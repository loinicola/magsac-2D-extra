#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "magsac_python.hpp"


namespace py = pybind11;

py::tuple adaptiveInlierSelection(
    py::array_t<double>  x1y1_,
    py::array_t<double>  x2y2_,
    py::array_t<double>  modelParameters_,
    double maximumThreshold_,
    int problemType_,
    int minimumInlierNumber_) 
{
    if (problemType_ < 0 || problemType_ > 2)
        throw std::invalid_argument("Variable 'problemType' should be in interval [0,2]");

    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (DIM != 2) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,2]");
    }

    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];

    if (DIMa != 2) {
        throw std::invalid_argument("x2y2 should be an array with dims [n,2]");
    }

    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument("x1y1 and x2y2 should be the same size");
    }

    py::buffer_info bufModel = modelParameters_.request();
    size_t DIMModelX = bufModel.shape[0];
    size_t DIMModelY = bufModel.shape[1];

    if (DIMModelX != 3 || DIMModelY != 3)
        throw std::invalid_argument("The model should be a 3*3 matrix.");

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);

    double* ptr1a = (double*)buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);

    double* ptrModel = (double*)bufModel.ptr;
    std::vector<double> modelParameters;
    modelParameters.assign(ptrModel, ptrModel + bufModel.size);

    std::vector<bool> inliers(NUM_TENTS);
    double bestThreshold;

    int inlierNumber = adaptiveInlierSelection_(
        x1y1,
        x2y2,
        modelParameters,
        inliers,
        bestThreshold,
        problemType_,
        maximumThreshold_,
        minimumInlierNumber_);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info bufInliers = inliers_.request();
    bool* ptrInliers = (bool*)bufInliers.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptrInliers[i] = inliers[i];

    return py::make_tuple(inliers_, inlierNumber, bestThreshold);

}

py::tuple findRobust2DRigidTransformation(
	py::array_t<double>  correspondences_,
	py::array_t<double>  probabilities_,
	int sampler,
    bool update_sampling,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    bool check_edges_similarity,
    double edges_similarity_threshold,
    bool check_min_edges_length,
    double min_edges_length_threshold,
    bool apply_rotation_boundaries,
    py::array_t<float> rotation_boundaries_,
    bool apply_translation_boundary,
    py::array_t<float> translation_init_,
    float translation_boundary,
    int core_num,
    int partition_num,
    bool reduced_matrix)
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("correspondences should be an array with dims [n,4], n>=2.");
	}
    if (NUM_TENTS < 2) {
        throw std::invalid_argument("correspondences should be an array with dims [n,4], n>=2.");
    }

    if (check_edges_similarity && (edges_similarity_threshold > 1.0 || edges_similarity_threshold <= 0.0)) {
        throw std::invalid_argument("edges_similarity_threshold should be >0 and <=1.");
    }

    if (check_min_edges_length && (min_edges_length_threshold <= 0.0)) {
        throw std::invalid_argument("min_edge_length_threshold should be >0.");
    }

    py::buffer_info buf_rotation_boundaries = rotation_boundaries_.request();
    size_t n_rotation_boundaries = buf_rotation_boundaries.shape[0];
    std::vector<float> rotation_boundaries;

    if (apply_rotation_boundaries) {
        if (n_rotation_boundaries != 2) {
            throw std::invalid_argument("rotation_boundaries should be [min,max] with min<max.");
        }
        float* ptr_rotation_boundaries = (float*)buf_rotation_boundaries.ptr;
        rotation_boundaries.assign(ptr_rotation_boundaries, ptr_rotation_boundaries + buf_rotation_boundaries.size);
        if (rotation_boundaries[0] >= rotation_boundaries[1]) {
            throw std::invalid_argument("rotation_boundaries should be [min,max] with min<max.");
        }
    } else {
        rotation_boundaries = { -M_PI_2, M_PI_2 };
    }

    if (apply_translation_boundary) {
        if (translation_boundary <= 0.0f) {
            throw std::invalid_argument("translation_boundary should be > 0.");
        }
    } else {
        translation_boundary = 999999.99f;
    }

    py::buffer_info buf_translation_init = translation_init_.request();
    size_t n_translation_init = buf_translation_init.shape[0];
    std::vector<float> translation_init;

    if (apply_translation_boundary) {
        if (n_translation_init != 2) {
            throw std::invalid_argument("translation_init should be [x,y].");
        }
        float* ptr_translation_init = (float*)buf_translation_init.ptr;
        translation_init.assign(ptr_translation_init, ptr_translation_init + buf_translation_init.size);
    } else {
        translation_init = { 0.0f, 0.0f};
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> T(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        if (buf_prob.size != NUM_TENTS) {
            throw std::invalid_argument("the selected sampler requires probabilities with same size as correspondences.");
        }
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    } else if (sampler == 2) {
        throw std::invalid_argument("the sampler Progressive NAPSAC is not compatible with rigid transformation estimation.");
    }

    int num_inl = findRobust2DRigidTransformation_(
        correspondences,
        inliers,
        T,
        probabilities,
        sampler,
        update_sampling,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        check_edges_similarity,
        edges_similarity_threshold,
        check_min_edges_length,
        min_edges_length_threshold,
        apply_rotation_boundaries,
        rotation_boundaries,
        apply_translation_boundary,
        translation_init,
        translation_boundary,
        core_num,
        partition_num);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
    }

    if (reduced_matrix) {
        py::array_t<double> T_ = py::array_t<double>({ 3,3 });
        py::buffer_info buf2 = T_.request();
        double* ptr2 = (double*)buf2.ptr;
        for (size_t i = 0; i < 9; i++)
            ptr2[i] = T[i];
        return py::make_tuple(T_, inliers_);

    } else {
        py::array_t<double> T_ = py::array_t<double>({ 4,4 });
        py::buffer_info buf2 = T_.request();
        double* ptr2 = (double*)buf2.ptr;
        ptr2[0] = T[0];  ptr2[1] = T[1];  ptr2[2] = 0;  ptr2[3] = T[2];
        ptr2[4] = T[3];  ptr2[5] = T[4];  ptr2[6] = 0;  ptr2[7] = T[5];
        ptr2[8] = 0;     ptr2[9] = 0;     ptr2[10] = 1; ptr2[11] = 0;
        ptr2[12] = T[6]; ptr2[13] = T[7]; ptr2[14] = 0; ptr2[15] = T[8];
        return py::make_tuple(T_, inliers_);
    }
}

py::tuple findRigidTransformation(
	py::array_t<double>  correspondences_,
	py::array_t<double>  probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 6) {
		throw std::invalid_argument("correspondences should be an array with dims [n,6], n>=3");
	}
    if (NUM_TENTS < 3) {
        throw std::invalid_argument("correspondences should be an array with dims [n,6], n>=3");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> T(16);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

    int num_inl = findRigidTransformation_(
        correspondences,
        inliers,
        T,
        probabilities,
        sampler,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        partition_num);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
    }
    py::array_t<double> T_ = py::array_t<double>({ 4,4 });
    py::buffer_info buf2 = T_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 16; i++)
        ptr2[i] = T[i];
    return py::make_tuple(T_, inliers_);
}

py::tuple findFundamentalMatrix(
	py::array_t<double>  correspondences_,
    double w1, 
    double h1,
    double w2,
    double h2,
	py::array_t<double>  probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
	}
    if (NUM_TENTS < 7) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> F(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

    int num_inl = findFundamentalMatrix_(
        correspondences,
        inliers,
        F,
        probabilities,
        w1,
        h1,
        w2,
        h2,
        sampler,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        partition_num);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
    }
    py::array_t<double> F_ = py::array_t<double>({ 3,3 });
    py::buffer_info buf2 = F_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = F[i];
    return py::make_tuple(F_, inliers_);
}

py::tuple findEssentialMatrix(
	py::array_t<double>  correspondences_,
    py::array_t<double>  K1_,
    py::array_t<double>  K2_,
    double w1, 
    double h1,
    double w2,
    double h2,
	py::array_t<double>  probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num) 
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=5");
	}
    if (NUM_TENTS < 5) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=5");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    py::buffer_info K1_buf = K1_.request();
    size_t three_a = K1_buf.shape[0];
    size_t three_b = K1_buf.shape[1];

    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument("K1 shape should be [3x3]");
    }
    double* ptr1_k = (double*)K1_buf.ptr;
    std::vector<double> K1;
    K1.assign(ptr1_k, ptr1_k + K1_buf.size);

    py::buffer_info K2_buf = K2_.request();
    three_a = K2_buf.shape[0];
    three_b = K2_buf.shape[1];

    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument("K2 shape should be [3x3]");
    }
    double* ptr2_k = (double*)K2_buf.ptr;
    std::vector<double> K2;
    K2.assign(ptr2_k, ptr2_k + K2_buf.size);

    std::vector<double> E(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

    int num_inl = findEssentialMatrix_(
        correspondences,
        inliers,
        E,
        K1, 
        K2,
        probabilities,
        w1,
        h1,
        w2,
        h2,
        sampler,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        partition_num);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
    }

    py::array_t<double> E_ = py::array_t<double>({ 3,3 });
    py::buffer_info buf2 = E_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = E[i];
    return py::make_tuple(E_, inliers_);
}
                                
py::tuple findHomography(
	                     py::array_t<double>  correspondences_,
                         double w1, 
                         double h1,
                         double w2,
                         double h2,
                         py::array_t<double>  probabilities_,
                         int sampler,
						 bool use_magsac_plus_plus,
                         double sigma_th,
                         double conf,
                         int min_iters,
                         int max_iters,
                         int partition_num) 
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=4");
	}
    if (NUM_TENTS < 4) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=4");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);
    
    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }
    
    int num_inl = findHomography_(
                    correspondences,
                    inliers,
                    H,
                    probabilities,
                    w1,
                    h1,
                    w2,
                    h2,
                    sampler,
					use_magsac_plus_plus,
                    sigma_th,
                    conf,
                    min_iters,
                    max_iters,
                    partition_num);
    
    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];   
    
    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> H_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = H_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = H[i];
    
    return py::make_tuple(H_,inliers_);
                         }
PYBIND11_PLUGIN(pymagsac_extra) {
                                                                             
    py::module m("pymagsac_extra", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pymagsac_extra
        .. autosummary::
           :toctree: _generate
           
           findEssentialMatrix,
           findFundamentalMatrix,
           findHomography,
           adaptiveInlierSelection
    
    )doc");

    m.def("adaptiveInlierSelection", &adaptiveInlierSelection, R"doc(some doc)doc",
        py::arg("x1y1"),
        py::arg("x2y2"),
        py::arg("modelParameters"),
        py::arg("maximumThreshold"),
        py::arg("problemType"),
        py::arg("minimumInlierNumber") = 20);

    m.def("findEssentialMatrix", &findEssentialMatrix, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("K1"),
        py::arg("K2"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

    m.def("findFundamentalMatrix", &findFundamentalMatrix, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

    m.def("findRobust2DRigidTransformation", &findRobust2DRigidTransformation, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("update_sampling") = false,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("check_edges_similarity") = false,
        py::arg("edges_similarity_threshold") = 0.9,
        py::arg("check_min_edges_length") = false,
        py::arg("min_edges_length_threshold") = 0.1,
        py::arg("apply_rotation_boundaries") = false,
        py::arg("rotation_boundaries") = py::array_t<float>(),
        py::arg("apply_translation_boundary") = false,
        py::arg("translation_init") = py::array_t<float>(),
        py::arg("translation_boundary") = 99999.99,
        py::arg("core_num") = 1,
        py::arg("partition_num") = 5,
        py::arg("reduced_matrix") = false);

    m.def("findRigidTransformation", &findRigidTransformation, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

  m.def("findHomography", &findHomography, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5); 


  return m.ptr();
}
