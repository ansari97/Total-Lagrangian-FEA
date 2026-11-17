#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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

  // Duration: --t_end (default) or --steps, but not both.
  if (cli.IsSet("t_end") && cli.IsSet("steps")) {
    std::cerr << "error: give either --t_end or --steps, not both\n";
    *exit_code = 1;
    return false;
  }
  if (opt->dt <= 0.0) {
    std::cerr << "error: --dt must be positive\n";
    *exit_code = 1;
    return false;
  }
  if (cli.IsSet("steps")) {
    opt->t_end = opt->steps * opt->dt;
  } else {
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
    std::cerr << "error: the run must have at least one time step\n";
    *exit_code = 1;
    return false;
  }

  if (opt->t_ramp < 0.0) {
    std::cerr << "error: --t_ramp must be >= 0\n";
    *exit_code = 1;
    return false;
  }

  return true;
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
            << " N, ramp = " << opt.t_ramp << " s\n";
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
int Run(const Options& opt) {
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

  for (int step = 1; step <= opt.steps; ++step) {
    const double t = step * opt.dt;  // time at the end of this step
    const double ramp =
        (opt.t_ramp > 0.0) ? std::min(1.0, t / opt.t_ramp) : 1.0;
    Fx = ramp * opt.Fx;
    Fz = ramp * opt.Fz;
    data.SetExternalForce(make_f_ext(Fx, Fz));
    solver.Solve();
    if (step % opt.print_every == 0 || step == opt.steps) {
      print_track(step);
    }
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
  PrintOptions(opt);
  try {
    return Run(opt);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}