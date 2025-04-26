#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nep.cpp"
#include "nep.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <tuple>

namespace py = pybind11;

// 计算列的平均值
std::vector<double> calculate_column_averages(const std::vector<std::vector<double>>& arr) {
    std::vector<double> averages;

    if (arr.empty()) return averages;

    size_t num_columns = arr[0].size();

    // 计算每列的平均值
    for (size_t col = 0; col < num_columns; ++col) {
        double sum = 0;
        size_t row_count = arr.size();
        for (size_t row = 0; row < row_count; ++row) {
            sum += arr[row][col];
        }
        averages.push_back(sum / row_count);
    }

    return averages;
}

// 计算行的平均值
std::vector<double> calculate_row_averages(const std::vector<std::vector<double>>& arr) {
    std::vector<double> averages;

    if (arr.empty()) return averages;

    // 遍历每一行
    for (const auto& row : arr) {
        double sum = 0;
        size_t num_elements = row.size();

        // 遍历当前行的每个元素，累加
        for (size_t i = 0; i < num_elements; ++i) {
            sum += row[i];
        }

        // 计算该行的平均值并保存
        averages.push_back(sum / num_elements);
    }

    return averages;
}

// 重塑数组（将一维数组重塑为二维）
void reshape(const std::vector<double>& input, int rows, int cols, std::vector<std::vector<double>>& result) {
    if (input.size() != rows * cols) {
        throw std::invalid_argument("The number of elements does not match the new shape.");
    }

    result.resize(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = input[i * cols + j];
        }
    }
}

// 矩阵转置
void transpose(const std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output) {
    int rows = input.size();
    int cols = input[0].size();

    // 初始化转置矩阵
    output.resize(cols, std::vector<double>(rows));

    // 执行转置操作
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[c][r] = input[r][c];
        }
    }
}

// 转换函数：UTF-8 到系统编码
std::string convert_path(const std::string& utf8_path) {
#ifdef _WIN32
    // Windows：将 UTF-8 转换为 ANSI（例如 GBK）
    int wstr_size = MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, nullptr, 0);
    std::wstring wstr(wstr_size, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, &wstr[0], wstr_size);

    int ansi_size = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string ansi_path(ansi_size, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &ansi_path[0], ansi_size, nullptr, nullptr);
    return ansi_path;
#else
    // Linux/macOS：直接返回 UTF-8
    return utf8_path;
#endif
}


class CpuNep : public NEP3 {
public:
    CpuNep(const std::string& potential_filename)  {


    std::string utf8_path  = convert_path(potential_filename);


    init_from_file(utf8_path, false);
    }

    // 计算所有结构的 potential, force, virial
void compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial)
{
  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }

  const int N = type.size();
  const int size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }

//  allocate_memory(N);
    int num_atoms = N;
    std::vector<double> sum_fxyz(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    std::vector<int> NN_radial(N);
    std::vector<int> NL_radial(N * MN);
    std::vector<int> NN_angular(N);
    std::vector<int> NL_angular(N * MN);
    std::vector<double> r12(N * MN * 6);
    std::vector<double> Fp(N * annmb.dim);
    int num_cells[3];
    double ebox[18];
  for (int n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (int n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (int n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial, paramb.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    true, false, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), potential.data(), nullptr, nullptr, nullptr);

  find_force_radial_small_box(
    false, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data());

  find_force_angular_small_box(
    false, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(),
    sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data());

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      N, paramb, zbl, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), force.data() + N,
      force.data() + N * 2, virial.data(), potential.data());
  }
}


std::tuple<std::vector<std::vector<double>>,
           std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
calculate(const std::vector<std::vector<int>>& type,
          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position) {

    size_t type_size = type.size();
    std::vector<std::vector<double>> potentials(type_size);  // 预分配空间
    std::vector<std::vector<double>> forces(type_size);      // 预分配空间
    std::vector<std::vector<double>> virials(type_size);     // 预分配空间


    // OpenMP 并行化报错
    #if defined(_OPENMP)
        #pragma omp parallel for
    #endif
    for (int i = 0; i < type_size; ++i) {

        potentials[i].resize(type[i].size());
        forces[i].resize(type[i].size() * 3);  // 假设 force 是 3D 向量
        virials[i].resize(type[i].size() * 9);  // 假设 virial 是 3x3 矩阵

        // 调用计算函数
        compute(type[i], box[i], position[i],
                potentials[i], forces[i], virials[i]);

    }

    return std::make_tuple(potentials, forces, virials);
}


    // 获取 descriptor
    std::vector<double> get_descriptor(const std::vector<int>& type,
                                       const std::vector<double>& box,
                                       const std::vector<double>& position) {
        std::vector<double> descriptor(type.size() * annmb.dim);
        find_descriptor(type, box, position, descriptor);
        return descriptor;
    }

    // 获取元素列表
    std::vector<std::string> get_element_list() {
        return element_list;
    }

    // 获取所有结构的 descriptor
    std::vector<std::vector<double>> get_structures_descriptor(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_descriptors(type_size, std::vector<double>(annmb.dim));

        for (int i = 0; i < type_size; ++i) {
            std::vector<double> struct_des(type[i].size() * annmb.dim);
            find_descriptor(type[i], box[i], position[i], struct_des);
//
            // 重塑 descriptor 以适应矩阵
            std::vector<std::vector<double>> struct_des_reshaped;
            reshape(struct_des, annmb.dim, type[i].size(), struct_des_reshaped);

            // 计算行平均
            all_descriptors[i] = calculate_row_averages(struct_des_reshaped);
        }

        return all_descriptors;
    }
    // 获取所有结构的 polarizability
    std::vector<std::vector<double>> get_structures_polarizability(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_polarizability(type_size, std::vector<double>(6));

        for (int i = 0; i < type_size; ++i) {
            std::vector<double> struct_pol(6);
            find_polarizability(type[i], box[i], position[i], struct_pol);

            all_polarizability[i] = struct_pol;
        }

        return all_polarizability;
    }

    // 获取所有结构的原子极化率polarizability
    std::vector<std::vector<double>> get_structures_polarizability_atomic(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {

        if (type.size() != 1 ){
            std::cout << "Can only calculate per-atom polarizability for one structure once.\n";
            exit(1);
        }
        std::vector<std::vector<double>> atomic_polarizability(type[0].size(), std::vector<double>(6));

        find_polarizability_atomic(type[0], box[0], position[0], atomic_polarizability);
        
        return atomic_polarizability;
    }
    
        // 获取所有结构的 dipole
    std::vector<std::vector<double>> get_structures_dipole(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_dipole(type_size, std::vector<double>(3));

        for (int i = 0; i < type_size; ++i) {
            std::vector<double> struct_dipole(3);
            find_dipole(type[i], box[i], position[i], struct_dipole);

            all_dipole[i] = struct_dipole;
        }

        return all_dipole;
    }
};

// pybind11 模块绑定
PYBIND11_MODULE(nep_cpu, m) {
    m.doc() = "A pybind11 module for NEP";

    py::class_<CpuNep>(m, "CpuNep")
        .def(py::init<const std::string&>(), py::arg("potential_filename"))
        .def("calculate", &CpuNep::calculate)
        .def("get_descriptor", &CpuNep::get_descriptor)

        .def("get_element_list", &CpuNep::get_element_list)
        .def("get_structures_polarizability", &CpuNep::get_structures_polarizability)
        .def("get_structures_polarizability_atomic", &CpuNep::get_structures_polarizability_atomic)
        .def("get_structures_dipole", &CpuNep::get_structures_dipole)

        .def("get_structures_descriptor", &CpuNep::get_structures_descriptor);

}
