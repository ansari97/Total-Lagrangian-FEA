#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/quadrature_utils.h"

namespace {

// Turn a path given on the command line into one that works under `bazel run`.
std::string ResolvePath(const std::string& path) {
  if (path.empty() || path[0] == '/') {
    return path;
  }
  const char* ws = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (ws == nullptr) {
    return path;
  }
  return std::string(ws) + "/" + path;
}

// Open a text file and return its non-empty, non-comment lines.
std::vector<std::string> ReadDataLines(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot open file: " + path);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

// Read a file with one node index per line (base, track, midplane files).
std::vector<int> ReadIndexFile(const std::string& path) {
  std::vector<int> indices;
  for (const std::string& line : ReadDataLines(path)) {
    indices.push_back(std::stoi(line));
  }
  return indices;
}

// Read the tip traction weights: one "node_index weight" pair per line.
void ReadWeightFile(const std::string& path, std::vector<int>* nodes,
                    std::vector<double>* weights) {
  for (const std::string& line : ReadDataLines(path)) {
    std::istringstream iss(line);
    int node;
    double w;
    if (!(iss >> node >> w)) {
      throw std::runtime_error("bad line in " + path + ": " + line);
    }
    nodes->push_back(node);
    weights->push_back(w);
  }
}

// A tip-force trajectory read from CSV: forces at times t (piecewise linear).
struct InputTrajectory {
  std::vector<double> t, Fx, Fz;
  bool empty() const {
    return t.empty();
  }
};

// Read "t,Fx,Fz" rows (an optional header line is skipped).
InputTrajectory ReadInputCsv(const std::string& path) {
  InputTrajectory in;
  for (const std::string& line : ReadDataLines(path)) {
    if (std::isalpha(static_cast<unsigned char>(line[0]))) {
      continue;  // header line such as "t,Fx,Fz"
    }
    std::string row = line;
    std::replace(row.begin(), row.end(), ',', ' ');
    std::istringstream iss(row);
    double t, fx, fz;
    if (!(iss >> t >> fx >> fz)) {
      throw std::runtime_error("bad line in " + path + ": " + line);
    }
    if (!in.t.empty() && t <= in.t.back()) {
      throw std::runtime_error("times must increase in " + path +
                               " at t = " + std::to_string(t));
    }
    in.t.push_back(t);
    in.Fx.push_back(fx);
    in.Fz.push_back(fz);
  }
  if (in.t.size() < 2) {
    throw std::runtime_error("input file needs at least two rows: " + path);
  }
  return in;
}

// Linear interpolation of the forces at time t (held constant outside the
// file's time range).
void InterpolateInput(const InputTrajectory& in, double t, double* Fx,
                      double* Fz) {
  if (t <= in.t.front()) {
    *Fx = in.Fx.front();
    *Fz = in.Fz.front();
    return;
  }
  if (t >= in.t.back()) {
    *Fx = in.Fx.back();
    *Fz = in.Fz.back();
    return;
  }
  const size_t k = std::upper_bound(in.t.begin(), in.t.end(), t) - in.t.begin();
  const double w = (t - in.t[k - 1]) / (in.t[k] - in.t[k - 1]);
  *Fx            = (1.0 - w) * in.Fx[k - 1] + w * in.Fx[k];
  *Fz            = (1.0 - w) * in.Fz[k - 1] + w * in.Fz[k];
}

// Copy the solver's current velocities (x, y, z per node) to the CPU.
Eigen::VectorXd GetVelocity(const SyncedNewtonSolver& solver, int n_nodes) {
  Eigen::VectorXd v(3 * n_nodes);
  const cudaError_t err =
      cudaMemcpy(v.data(), solver.GetVelocityGuessDevicePtr(),
                 3 * n_nodes * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("velocity copy failed: ") +
                             cudaGetErrorString(err));
  }
  return v;
}

// Write positions and velocities: one line per node, "i x y z vx vy vz".
void WriteState(const std::string& path, const Eigen::VectorXd& x,
                const Eigen::VectorXd& y, const Eigen::VectorXd& z,
                const Eigen::VectorXd& v) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  out << "# finger state: node x y z vx vy vz (SI units)\n";
  out << std::setprecision(17);
  for (int i = 0; i < x.size(); ++i) {
    out << i << " " << x(i) << " " << y(i) << " " << z(i) << " " << v(3 * i)
        << " " << v(3 * i + 1) << " " << v(3 * i + 2) << "\n";
  }
}

// Read a state written by WriteState (must match the mesh's node count).
void ReadState(const std::string& path, int n_nodes, Eigen::VectorXd* x,
               Eigen::VectorXd* y, Eigen::VectorXd* z, Eigen::VectorXd* v) {
  const std::vector<std::string> lines = ReadDataLines(path);
  if (static_cast<int>(lines.size()) != n_nodes) {
    throw std::runtime_error("state file " + path + " has " +
                             std::to_string(lines.size()) +
                             " nodes, mesh has " + std::to_string(n_nodes));
  }
  x->resize(n_nodes);
  y->resize(n_nodes);
  z->resize(n_nodes);
  v->resize(3 * n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    std::istringstream iss(lines[i]);
    int node;
    double px, py, pz, vx, vy, vz;
    if (!(iss >> node >> px >> py >> pz >> vx >> vy >> vz) || node != i) {
      throw std::runtime_error("bad line in " + path + ": " + lines[i]);
    }
    (*x)(i)         = px;
    (*y)(i)         = py;
    (*z)(i)         = pz;
    (*v)(3 * i)     = vx;
    (*v)(3 * i + 1) = vy;
    (*v)(3 * i + 2) = vz;
  }
}

// Streams rows of doubles into a NumPy .npy file (format 1.0, little-endian
// float64, C order). The header has a fixed size so the row count can be
// corrected at Close() if the run ends early.
class NpyWriter {
 public:
  // shape = (n_rows, n_cols); with n_cols == 0 the array is 1-D: (n_rows,).
  NpyWriter(const std::string& path, long n_rows, long n_cols)
      : path_(path), n_rows_(n_rows), n_cols_(n_cols) {
    out_.open(path, std::ios::binary);
    if (!out_) {
      throw std::runtime_error("cannot write " + path);
    }
    WriteHeader(n_rows_);
  }

  ~NpyWriter() {
    Close();
  }

  // Append one row (n_cols values; one value for a 1-D array).
  void AppendRow(const double* values) {
    const long n = (n_cols_ == 0) ? 1 : n_cols_;
    out_.write(reinterpret_cast<const char*>(values), n * sizeof(double));
    ++rows_written_;
  }
  void AppendRow(const Eigen::VectorXd& row) {
    if (row.size() != ((n_cols_ == 0) ? 1 : n_cols_)) {
      throw std::runtime_error("row size mismatch in " + path_);
    }
    AppendRow(row.data());
  }
  void AppendValue(double value) {
    AppendRow(&value);
  }

  // Finish the file; fix the row count in the header if it changed.
  void Close() {
    if (!out_.is_open())
      return;
    if (rows_written_ != n_rows_) {
      out_.seekp(0);
      WriteHeader(rows_written_);
    }
    out_.close();
  }

 private:
  void WriteHeader(long rows) {
    std::string dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) +
                       (n_cols_ == 0 ? "," : ", " + std::to_string(n_cols_)) +
                       "), }";
    const size_t kHeaderTotal =
        128;  // magic(6) + version(2) + length(2) + dict
    dict.append(kHeaderTotal - 10 - dict.size() - 1, ' ');
    dict.push_back('\n');
    const uint16_t len = static_cast<uint16_t>(dict.size());
    out_.write("\x93NUMPY", 6);
    out_.put(1);
    out_.put(0);
    out_.write(reinterpret_cast<const char*>(&len), 2);
    out_.write(dict.data(), dict.size());
  }

  std::string path_;
  std::ofstream out_;
  long n_rows_, n_cols_;
  long rows_written_ = 0;
};

// Write a whole vector as a 1-D .npy file.
void SaveVectorNpy(const std::string& path, const Eigen::VectorXd& v) {
  NpyWriter w(path, v.size(), 0);
  for (int k = 0; k < v.size(); ++k)
    w.AppendValue(v(k));
}

// Write the scalar mass matrix in "row col value" (COO) text form.
void WriteMassCoo(const std::string& path, const std::vector<int>& offsets,
                  const std::vector<int>& cols,
                  const std::vector<double>& vals) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  out << "# row col value (scalar mass matrix m_ij, 0-based)\n";
  out << std::setprecision(17);
  for (size_t i = 0; i + 1 < offsets.size(); ++i) {
    for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
      out << i << " " << cols[k] << " " << vals[k] << "\n";
    }
  }
}

// Stack per-node x, y, z into one vector [x0, y0, z0, x1, y1, z1, ...].
Eigen::VectorXd Interleave(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                           const Eigen::VectorXd& z) {
  Eigen::VectorXd e(3 * x.size());
  for (int i = 0; i < x.size(); ++i) {
    e(3 * i)     = x(i);
    e(3 * i + 1) = y(i);
    e(3 * i + 2) = z(i);
  }
  return e;
}

// All run settings in one place. Defaults: Dragon Skin 30 finger, dev mesh.
struct Options {
  std::string mesh_stem =
      "soft_finger/meshes/finger_100x15x15mm_n20x2x2/"
      "finger_100x15x15mm_n20x2x2";
  double young    = 1.0e6;   // Young's modulus Y (Pa)
  double poisson  = 0.45;    // Poisson's ratio nu (-)
  double density  = 1080.0;  // reference density rho_r (kg/m^3)
  double mu_v     = 750.0;   // Kelvin-Voigt viscosity mu_v (Pa s)
  double lambda_v = 6.7e3;   // Kelvin-Voigt viscosity lambda_v (Pa s)
  bool gravity    = true;    // include f^bdy
  double g0       = 9.81;    // gravity magnitude (m/s^2), acts along -z
  double dt       = 2e-4;    // time step h (s)
  double t_end    = 0.1;     // end time T (s)
  int steps       = 500;     // number of time steps (= T / h)
  int print_every = 50;      // print the tracking node every N steps

  std::string dump_mass;

  double Fx     = 0.0;  // constant tip force, x-component (N)
  double Fz     = 0.0;  // constant tip force, z-component (N)
  double t_ramp = 0.0;  // ramp the force up linearly over this time (s)

  std::string input;  // CSV file with columns t,Fx,Fz (empty = constant force)
  bool t_end_set = false;  // was --t_end given?
  bool steps_set = false;  // was --steps given?

  std::string init_state;  // start from this saved state (empty = reference)
  std::string save_state;  // write the final state here (empty = don't)

  std::string
      out_dir;  // write snapshot data into this folder (empty = no data)
  int save_every = 10;     // save every k-th step
  bool save_fint = false;  // also save internal-force snapshots
};

// Fill `opt` from the command line. Returns false if the program should stop,
// with the exit code in *exit_code (0 after --help, 1 after an error).
bool ParseOptions(int argc, char** argv, Options* opt, int* exit_code) {
  ANCFCPUUtils::Cli cli("finger_fom");
  cli.SetDescription(
      "TL-FEA full-order model of a clamped soft finger (T10, SVK + "
      "Kelvin-Voigt).");
  cli.AddString("mesh", opt->mesh_stem, "mesh path without extension");
  cli.AddDouble("young", opt->young, "Young's modulus Y (Pa)");
  cli.AddDouble("poisson", opt->poisson, "Poisson's ratio nu");
  cli.AddDouble("density", opt->density, "reference density rho_r (kg/m^3)");
  cli.AddDouble("mu_v", opt->mu_v, "Kelvin-Voigt viscosity mu_v (Pa s)");
  cli.AddDouble("lambda_v", opt->lambda_v,
                "Kelvin-Voigt viscosity lambda_v (Pa s)");
  cli.AddBool("gravity", opt->gravity, "include gravity");
  cli.AddDouble("g0", opt->g0, "gravity magnitude (m/s^2), along -z");
  cli.AddDouble("dt", opt->dt, "time step h (s)");
  cli.AddDouble("t_end", opt->t_end, "end time T (s); steps = T / dt");
  cli.AddInt("steps", opt->steps,
             "number of time steps (alternative to --t_end)");
  cli.AddInt("print_every", opt->print_every,
             "print tracking node every N steps");
  cli.AddString(
      "dump_mass", opt->dump_mass,
      "write the scalar mass matrix (row col value) to FILE and stop");
  cli.AddDouble("Fx", opt->Fx, "constant tip force, x-component (N)");
  cli.AddDouble("Fz", opt->Fz, "constant tip force, z-component (N)");
  cli.AddDouble(
      "t_ramp", opt->t_ramp,
      "ramp the force linearly from 0 over this time (s); 0 = sudden");
  cli.AddString("input", opt->input, "CSV file with columns t,Fx,Fz");
  cli.AddString("init_state", opt->init_state, "start from a saved state file");
  cli.AddString("save_state", opt->save_state, "write the final state to FILE");
  cli.AddString("out", opt->out_dir, "write snapshot data into this folder");
  cli.AddInt("save_every", opt->save_every, "save every k-th step");
  cli.AddBool("save_fint", opt->save_fint,
              "also save internal-force snapshots");

  std::string err;
  if (!cli.Parse(argc, argv, &err)) {
    std::cerr << "error: " << err << "\n";
    cli.PrintUsage(std::cerr);
    *exit_code = 1;
    return false;
  }
  if (cli.HelpRequested()) {
    cli.PrintUsage(std::cout);
    *exit_code = 0;
    return false;
  }

  opt->mesh_stem   = cli.GetString("mesh");
  opt->young       = cli.GetDouble("young");
  opt->poisson     = cli.GetDouble("poisson");
  opt->density     = cli.GetDouble("density");
  opt->mu_v        = cli.GetDouble("mu_v");
  opt->lambda_v    = cli.GetDouble("lambda_v");
  opt->gravity     = cli.GetBool("gravity");
  opt->g0          = cli.GetDouble("g0");
  opt->dt          = cli.GetDouble("dt");
  opt->t_end       = cli.GetDouble("t_end");
  opt->steps       = cli.GetInt("steps");
  opt->print_every = cli.GetInt("print_every");
  opt->dump_mass   = cli.GetString("dump_mass");
  opt->Fx          = cli.GetDouble("Fx");
  opt->Fz          = cli.GetDouble("Fz");
  opt->t_ramp      = cli.GetDouble("t_ramp");

  opt->input     = cli.GetString("input");
  opt->t_end_set = cli.IsSet("t_end");
  opt->steps_set = cli.IsSet("steps");

  opt->init_state = cli.GetString("init_state");
  opt->save_state = cli.GetString("save_state");

  opt->out_dir    = cli.GetString("out");
  opt->save_every = cli.GetInt("save_every");
  opt->save_fint  = cli.GetBool("save_fint");

  auto fail = [&](const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    *exit_code = 1;
    return false;
  };
  if (opt->t_end_set && opt->steps_set) {
    return fail("give either --t_end or --steps, not both");
  }
  if (opt->dt <= 0.0) {
    return fail("--dt must be positive");
  }
  if (opt->t_ramp < 0.0) {
    return fail("--t_ramp must be >= 0");
  }
  if (!opt->input.empty() &&
      (cli.IsSet("Fx") || cli.IsSet("Fz") || cli.IsSet("t_ramp"))) {
    return fail("--input replaces --Fx/--Fz/--t_ramp; do not combine them");
  }
  if (opt->save_every < 1) {
    return fail("--save_every must be >= 1");
  }
  return true;
}

// Decide the number of steps from --steps, --t_end, or the input file.
void FinalizeDuration(Options* opt, const InputTrajectory& in) {
  if (opt->steps_set) {
    opt->t_end = opt->steps * opt->dt;
  } else {
    if (!opt->t_end_set && !in.empty()) {
      opt->t_end = in.t.back();  // default: run to the end of the file
    }
    opt->steps            = static_cast<int>(std::lround(opt->t_end / opt->dt));
    const double t_actual = opt->steps * opt->dt;
    if (std::abs(t_actual - opt->t_end) > 1e-9 * opt->t_end) {
      std::cout << "note: t_end = " << opt->t_end
                << " s is not a multiple of dt; using " << opt->steps
                << " steps (T = " << t_actual << " s)\n";
    }
    opt->t_end = t_actual;
  }
  if (opt->steps < 1) {
    throw std::runtime_error("the run must have at least one time step");
  }
  if (!in.empty()) {
    if (opt->t_end > in.t.back() * (1.0 + 1e-12)) {
      throw std::runtime_error("run (T = " + std::to_string(opt->t_end) +
                               " s) is longer than the input file (ends at " +
                               std::to_string(in.t.back()) + " s)");
    }
    double max_gap = 0.0;
    for (size_t k = 1; k < in.t.size(); ++k) {
      max_gap = std::max(max_gap, in.t[k] - in.t[k - 1]);
    }
    if (max_gap > opt->dt * (1.0 + 1e-9)) {
      std::cout << "warning: input is sampled more coarsely (" << max_gap
                << " s) than dt (" << opt->dt << " s)\n";
    }
  }
}

// Print the settings so every run records what it used.
void PrintOptions(const Options& opt) {
  const double mu  = opt.young / (2.0 * (1.0 + opt.poisson));
  const double lam = opt.young * opt.poisson /
                     ((1.0 + opt.poisson) * (1.0 - 2.0 * opt.poisson));
  std::cout << "mesh      : " << opt.mesh_stem << "\n"
            << "material  : Y = " << opt.young << " Pa, nu = " << opt.poisson
            << ", rho_r = " << opt.density << " kg/m^3\n"
            << "  Lame    : mu = " << mu << " Pa, lambda = " << lam << " Pa\n"
            << "  KV      : mu_v = " << opt.mu_v
            << " Pa s, lambda_v = " << opt.lambda_v << " Pa s\n"
            << "gravity   : " << (opt.gravity ? "on" : "off")
            << ", g0 = " << opt.g0 << " m/s^2\n"
            << "time      : h = " << opt.dt << " s, steps = " << opt.steps
            << " (T = " << opt.t_end << " s)\n"
            << "tip force : Fx = " << opt.Fx << " N, Fz = " << opt.Fz
            << " N, ramp = " << opt.t_ramp << " s\n"
            << "input     : "
            << (opt.input.empty() ? "constant force" : opt.input) << "\n"
            << "initial   : "
            << (opt.init_state.empty() ? "reference (undeformed)"
                                       : opt.init_state)
            << "\n"
            << "output    : " << (opt.out_dir.empty() ? "none" : opt.out_dir)
            << (opt.out_dir.empty()
                    ? ""
                    : "  (every " + std::to_string(opt.save_every) + " steps" +
                          (opt.save_fint ? ", with f_int)" : ")"))
            << "\n";
}

// Everything read from the mesh files.
struct MeshData {
  Eigen::MatrixXd nodes;            // n_nodes x 3 reference coordinates (m)
  Eigen::MatrixXi elements;         // n_elems x 10 node indices (0-based)
  std::vector<int> base;            // clamped base nodes
  std::vector<int> tip_nodes;       // tip-face nodes with nonzero weight
  std::vector<double> tip_weights;  // traction weights beta_i (sum to 1)
  int track = -1;                   // tracking node (tip-face center)
};

// Read the mesh and its companion files, then check they fit together.
MeshData LoadMesh(const std::string& mesh_stem) {
  const std::string stem = ResolvePath(mesh_stem);
  MeshData mesh;

  const int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes(stem + ".1.node", mesh.nodes);
  const int n_elems =
      ANCFCPUUtils::FEAT10_read_elements(stem + ".1.ele", mesh.elements);
  if (n_nodes <= 0 || n_elems <= 0) {
    throw std::runtime_error("mesh reading failed for " + stem);
  }

  mesh.base = ReadIndexFile(stem + ".base.txt");
  ReadWeightFile(stem + ".tip_weights.txt", &mesh.tip_nodes, &mesh.tip_weights);
  const std::vector<int> track = ReadIndexFile(stem + ".track.txt");
  if (track.size() != 1) {
    throw std::runtime_error("track file must contain exactly one node");
  }
  mesh.track = track[0];

  // Every index must refer to an existing node.
  auto check_index = [&](int i, const char* what) {
    if (i < 0 || i >= n_nodes) {
      throw std::runtime_error(std::string(what) +
                               " index out of range: " + std::to_string(i));
    }
  };
  for (int i : mesh.base)
    check_index(i, "base node");
  for (int i : mesh.tip_nodes)
    check_index(i, "tip node");
  check_index(mesh.track, "tracking node");

  // The tip weights must add up to 1 (resultant = applied force).
  double weight_sum = 0.0;
  for (double w : mesh.tip_weights)
    weight_sum += w;
  if (std::abs(weight_sum - 1.0) > 1e-10) {
    throw std::runtime_error("tip weights do not sum to 1");
  }

  std::cout << "mesh      : " << n_nodes << " nodes, " << n_elems
            << " T10 elements, " << mesh.base.size() << " clamped nodes, "
            << mesh.tip_nodes.size() << " loaded tip nodes\n"
            << "tracking  : node " << mesh.track << " at ("
            << mesh.nodes(mesh.track, 0) << ", " << mesh.nodes(mesh.track, 1)
            << ", " << mesh.nodes(mesh.track, 2) << ")\n";
  return mesh;
}

// The whole simulation. Later blocks add their steps here.
int Run(const Options& opt, const InputTrajectory& input) {
  const MeshData mesh = LoadMesh(opt.mesh_stem);

  const int n_nodes = static_cast<int>(mesh.nodes.rows());
  const int n_elems = static_cast<int>(mesh.elements.rows());

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "GPU       : " << prop.name << " (cc " << prop.major << "."
            << prop.minor << ")\n";

  GPU_FEAT10_Data data(n_elems, n_nodes);
  data.Initialize();

  Eigen::VectorXi clamped(static_cast<int>(mesh.base.size()));
  for (int k = 0; k < clamped.size(); ++k) {
    clamped(k) = mesh.base[k];
  }
  FEAT10ConstraintManager constraints(&data);
  constraints.AddNodesToWorldCD(clamped);
  constraints.Finalize();

  data.SetExternalForce(Eigen::VectorXd::Zero(3 * n_nodes));

  const Eigen::VectorXd ref_x = mesh.nodes.col(0);
  const Eigen::VectorXd ref_y = mesh.nodes.col(1);
  const Eigen::VectorXd ref_z = mesh.nodes.col(2);
  data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
             Quadrature::tet5pt_weights, ref_x, ref_y, ref_z, mesh.elements);

  const SolidMaterialProperties material = SolidMaterialProperties::SVK(
      opt.young, opt.poisson, opt.density, opt.mu_v, opt.lambda_v);
  data.ApplyMaterial(material);

  data.CalcDnDuPre();
  data.CalcMassMatrix();
  data.CalcConstraintData();
  data.ConvertToCSR_ConstraintJacT();
  data.BuildConstraintJacobianCSR();
  data.CalcP();
  data.CalcInternalForce();

  std::vector<int> mass_offsets, mass_cols;
  std::vector<double> mass_vals;
  data.RetrieveMassCSRToCPU(mass_offsets, mass_cols, mass_vals);
  double total_mass = 0.0;
  for (double v : mass_vals)
    total_mass += v;

  Eigen::VectorXd f_int;
  data.RetrieveInternalForceToCPU(f_int);

  std::cout << "model     : " << 3 * n_nodes << " coordinates, "
            << data.get_n_constraint() << " constraint equations, "
            << mass_vals.size() << " mass nonzeros\n"
            << "checks    : total mass = " << total_mass
            << " kg, max |f_int| at reference = " << f_int.cwiseAbs().maxCoeff()
            << " N\n";

  if (!opt.dump_mass.empty()) {
    const std::string path = ResolvePath(opt.dump_mass);
    std::ofstream out(path);
    if (!out) {
      throw std::runtime_error("cannot write " + path);
    }
    out << "# row col value (scalar mass matrix m_ij, 0-based)\n";
    out << std::setprecision(17);
    for (int i = 0; i < n_nodes; ++i) {
      for (int k = mass_offsets[i]; k < mass_offsets[i + 1]; ++k) {
        out << i << " " << mass_cols[k] << " " << mass_vals[k] << "\n";
      }
    }
    std::cout << "wrote mass matrix (" << mass_vals.size() << " nonzeros) to "
              << path << "\n";
    return 0;
  }

  Eigen::VectorXd f_bdy = Eigen::VectorXd::Zero(3 * n_nodes);
  if (opt.gravity) {
    for (int i = 0; i < n_nodes; ++i) {
      double row_sum = 0.0;
      for (int k = mass_offsets[i]; k < mass_offsets[i + 1]; ++k) {
        row_sum += mass_vals[k];
      }
      f_bdy(3 * i + 2) = -opt.g0 * row_sum;
    }
  }

  Eigen::VectorXd b_x = Eigen::VectorXd::Zero(3 * n_nodes);
  Eigen::VectorXd b_z = Eigen::VectorXd::Zero(3 * n_nodes);
  for (size_t k = 0; k < mesh.tip_nodes.size(); ++k) {
    const int i    = mesh.tip_nodes[k];
    b_x(3 * i + 0) = mesh.tip_weights[k];
    b_z(3 * i + 2) = mesh.tip_weights[k];
  }

  auto make_f_ext = [&](double Fx, double Fz) -> Eigen::VectorXd {
    return f_bdy + Fx * b_x + Fz * b_z;
  };

  std::cout << "forces    : total gravity z = " << f_bdy.sum()
            << " N, sum(b_x) = " << b_x.sum() << ", sum(b_z) = " << b_z.sum()
            << "\n";

  SyncedNewtonParams params;
  params.inner_atol = 1e-9;    // Newton stops when ||g|| < 1e-9 N ...
  params.inner_rtol = 1e-10;   // ... or ||g|| <= 1e-10 * (||g|| at start)
  params.outer_tol  = 1e-10;   // multiplier loop stops when ||c|| < 1e-10 m
  params.rho        = 1e10;    // augmented-Lagrangian penalty rho
  params.max_outer  = 10;      // at most 10 multiplier updates per step
  params.max_inner  = 20;      // at most 20 Newton iterations per update
  params.time_step  = opt.dt;  // h
  params.enable_line_search = false;  // take full Newton steps

  SyncedNewtonSolver solver(&data, data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  Eigen::VectorXd v_prev = Eigen::VectorXd::Zero(3 * n_nodes);

  if (!opt.init_state.empty()) {
    Eigen::VectorXd x0, y0, z0, v0;
    ReadState(ResolvePath(opt.init_state), n_nodes, &x0, &y0, &z0, &v0);
    data.UpdatePositions(x0, y0, z0);
    solver.SetInitialVelocity(v0);
    v_prev = v0;
    std::cout << "initial   : state read from " << opt.init_state << "\n";
  }

  const bool saving = !opt.out_dir.empty();
  const int n_save  = opt.steps / opt.save_every;
  std::unique_ptr<NpyWriter> w_t, w_e, w_v, w_a, w_u, w_fint, w_fint_e;
  std::string out;
  if (saving) {
    out = ResolvePath(opt.out_dir);
    std::filesystem::create_directories(out);

    Eigen::VectorXd x0, y0, z0;
    data.RetrievePositionToCPU(x0, y0, z0);
    SaveVectorNpy(out + "/e_ref.npy", Interleave(ref_x, ref_y, ref_z));
    SaveVectorNpy(out + "/e0.npy", Interleave(x0, y0, z0));
    SaveVectorNpy(out + "/v0.npy", v_prev);
    SaveVectorNpy(out + "/fbdy.npy", f_bdy);
    {
      NpyWriter w(out + "/B.npy", 3 * n_nodes, 2);
      for (int k = 0; k < 3 * n_nodes; ++k) {
        const double row[2] = {b_x(k), b_z(k)};
        w.AppendRow(row);
      }
    }
    WriteMassCoo(out + "/mass_coo.txt", mass_offsets, mass_cols, mass_vals);

    const long nd = 3L * n_nodes;
    w_t.reset(new NpyWriter(out + "/t.npy", n_save, 0));
    w_e.reset(new NpyWriter(out + "/e.npy", n_save, nd));
    w_v.reset(new NpyWriter(out + "/v.npy", n_save, nd));
    w_a.reset(new NpyWriter(out + "/a.npy", n_save, nd));
    w_u.reset(new NpyWriter(out + "/u.npy", n_save, 2));
    if (opt.save_fint) {
      w_fint.reset(new NpyWriter(out + "/fint.npy", n_save, nd));
      w_fint_e.reset(new NpyWriter(out + "/fint_e.npy", n_save, nd));
    }
  }

  Eigen::VectorXd x, y, z;
  double Fx = 0.0, Fz = 0.0;  // current tip forces (N)
  auto print_track = [&](int step) {
    data.RetrievePositionToCPU(x, y, z);
    const int t = mesh.track;
    std::cout << "track  step " << step << ",\t" << "t = " << step * opt.dt
              << " s" << ",\t"
              << "pos = (" << x(t) << ", " << y(t) << ", " << z(t) << ")"
              << ",\t" << "F = (" << Fx << ", " << Fz << ")" << ",\t"
              << "disp = (" << x(t) - mesh.nodes(t, 0) << ", "
              << y(t) - mesh.nodes(t, 1) << ", " << z(t) - mesh.nodes(t, 2)
              << ")\n";
  };

  const auto t_start = std::chrono::steady_clock::now();

  for (int step = 1; step <= opt.steps; ++step) {
    const double t = step * opt.dt;  // time at the end of this step
    if (!input.empty()) {
      InterpolateInput(input, t, &Fx, &Fz);
    } else {
      const double ramp =
          (opt.t_ramp > 0.0) ? std::min(1.0, t / opt.t_ramp) : 1.0;
      Fx = ramp * opt.Fx;
      Fz = ramp * opt.Fz;
    }
    data.SetExternalForce(make_f_ext(Fx, Fz));
    solver.Solve();
    if (saving) {
      const Eigen::VectorXd v = GetVelocity(solver, n_nodes);
      if (step % opt.save_every == 0) {
        data.RetrievePositionToCPU(x, y, z);
        const Eigen::VectorXd a = (v - v_prev) / opt.dt;
        const double u[2]       = {Fx, Fz};
        w_t->AppendValue(t);
        w_e->AppendRow(Interleave(x, y, z));
        w_v->AppendRow(v);
        w_a->AppendRow(a);
        w_u->AppendRow(u);
        if (opt.save_fint) {
          Eigen::VectorXd f_tot, f_sum;
          data.RetrieveInternalForceToCPU(f_tot);
          data.CalcP();
          data.CalcInternalForce();
          data.RetrieveInternalForceToCPU(f_sum);
          w_fint->AppendRow(f_tot);
          w_fint_e->AppendRow(f_sum - f_tot);
        }
      }
      v_prev = v;
    }
    if (step % opt.print_every == 0 || step == opt.steps) {
      print_track(step);
    }
  }

  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
          .count();

  if (saving) {
    for (auto* w : {w_t.get(), w_e.get(), w_v.get(), w_a.get(), w_u.get(),
                    w_fint.get(), w_fint_e.get()}) {
      if (w != nullptr)
        w->Close();
    }
    std::ofstream info(out + "/info.json");
    info << std::setprecision(17) << "{\n"
         << "  \"mesh\": \"" << opt.mesh_stem << "\",\n"
         << "  \"n_nodes\": " << n_nodes << ",\n"
         << "  \"n_dofs\": " << 3 * n_nodes << ",\n"
         << "  \"young\": " << opt.young << ",\n"
         << "  \"poisson\": " << opt.poisson << ",\n"
         << "  \"density\": " << opt.density << ",\n"
         << "  \"mu_v\": " << opt.mu_v << ",\n"
         << "  \"lambda_v\": " << opt.lambda_v << ",\n"
         << "  \"gravity\": " << (opt.gravity ? "true" : "false") << ",\n"
         << "  \"g0\": " << opt.g0 << ",\n"
         << "  \"dt\": " << opt.dt << ",\n"
         << "  \"steps\": " << opt.steps << ",\n"
         << "  \"t_end\": " << opt.t_end << ",\n"
         << "  \"save_every\": " << opt.save_every << ",\n"
         << "  \"n_saved\": " << n_save << ",\n"
         << "  \"save_fint\": " << (opt.save_fint ? "true" : "false") << ",\n"
         << "  \"input\": \"" << opt.input << "\",\n"
         << "  \"Fx\": " << opt.Fx << ",\n"
         << "  \"Fz\": " << opt.Fz << ",\n"
         << "  \"t_ramp\": " << opt.t_ramp << ",\n"
         << "  \"init_state\": \"" << opt.init_state << "\",\n"
         << "  \"wall_time_s\": " << wall << ",\n"
         << "  \"layout\": \"rows = saved steps; columns = [x0,y0,z0,x1,...]; "
            "all quantities at t_n (end of step n); a = (v_n - v_{n-1})/dt\"\n"
         << "}\n";
    std::cout << "data      : " << n_save << " snapshots written to "
              << opt.out_dir << " (" << wall << " s)\n";
  }

  if (!opt.save_state.empty()) {
    data.RetrievePositionToCPU(x, y, z);
    const Eigen::VectorXd v_end = GetVelocity(solver, n_nodes);
    WriteState(ResolvePath(opt.save_state), x, y, z, v_end);

    // Static-equilibrium check on free nodes: elastic force vs. applied force.
    Eigen::VectorXd f_tot, f_sum;
    data.RetrieveInternalForceToCPU(
        f_tot);                // solver's total: elastic + viscous
    data.CalcP();              // stress without velocity
    data.CalcInternalForce();  // ADDS the elastic force to the array
    data.RetrieveInternalForceToCPU(f_sum);
    const Eigen::VectorXd f_el = f_sum - f_tot;  // elastic force alone

    const Eigen::VectorXd f_end = make_f_ext(Fx, Fz);
    std::vector<bool> is_clamped(n_nodes, false);
    for (int i : mesh.base)
      is_clamped[i] = true;
    double max_res = 0.0, max_v = 0.0;
    for (int i = 0; i < n_nodes; ++i) {
      if (is_clamped[i])
        continue;
      for (int c = 0; c < 3; ++c) {
        max_res =
            std::max(max_res, std::abs(f_el(3 * i + c) - f_end(3 * i + c)));
        max_v = std::max(max_v, std::abs(v_end(3 * i + c)));
      }
    }
    std::cout << "saved     : final state to " << opt.save_state
              << "  (free nodes: max|v| = " << max_v
              << " m/s, max|f_int,e - f_ext| = " << max_res << " N)\n";
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  int exit_code = 0;
  if (!ParseOptions(argc, argv, &opt, &exit_code)) {
    return exit_code;
  }
  try {
    InputTrajectory input;
    if (!opt.input.empty()) {
      input = ReadInputCsv(ResolvePath(opt.input));
    }
    FinalizeDuration(&opt, input);
    PrintOptions(opt);
    return Run(opt, input);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}