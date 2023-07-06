// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

#include "solver_rigid_transformation_svd.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class Robust2DRigidTransformationEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

			// A flag to decide if the edges similarity should be checked
			bool check_edges_similarity;
			// Threshold to validate the sample based on the similarity of the edges between the points
			double edges_similarity_threshold;
			double squared_edges_similarity_threshold;

			// A flag to decide if the minimum edge length should be checked
			bool check_min_edges_length;
			// Threshold to validate the sample based on the minimum edge length
			double min_edges_length_threshold;
			double squared_min_edges_length_threshold;

			// A flag to decide if the model should be checked against defined rotation boundaries
			bool apply_rotation_boundaries;
			// Rotation boundaries to validate the model
			std::vector<float> rotation_boundaries;
			float normalized_rotation_boundaries_range;

			// A flag to decide if the model should be checked against defined translation boundary
			bool apply_translation_boundary;
			// Translation boundary to validate the model
			std::vector<float> translation_init;
			float translation_boundary;
			float squared_translation_boundary;
			

		public:
			Robust2DRigidTransformationEstimator() :
				minimal_solver(std::make_shared<_MinimalSolverEngine>()), // Minimal solver engine used for estimating a model from a minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>()) // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			{}
			~Robust2DRigidTransformationEstimator() {}

			// A function to set the active state of the edges similarity threshold
			void setCheckEdgesSimilarity(const bool check_edges_similarity_) {
				check_edges_similarity = check_edges_similarity_;
			}

			// A function to set the edges similarity threshold 
			void setEdgesSimilarityThreshold(const double edges_similarity_threshold_) 
			{
				edges_similarity_threshold = edges_similarity_threshold_;
				squared_edges_similarity_threshold = edges_similarity_threshold * edges_similarity_threshold;
			}

			void setCheckMinEdgesLength(const bool check_min_edges_length_) {
				check_min_edges_length = check_min_edges_length_;
			}

			void setMinEdgesLengthThreshold(const double min_edges_length_threshold_) {
				min_edges_length_threshold = min_edges_length_threshold_;
				squared_min_edges_length_threshold = min_edges_length_threshold * min_edges_length_threshold;
			}

			void setApplyRotationBoundaries(const bool apply_rotation_boundaries_) {
				apply_rotation_boundaries = apply_rotation_boundaries_;
			}

			void setRotationBoundaries(const std::vector<float> rotation_boundaries_) {
				rotation_boundaries = rotation_boundaries_;
				normalized_rotation_boundaries_range = fmod(rotation_boundaries[1] - rotation_boundaries[0], 2 * M_PIf32);
				if (normalized_rotation_boundaries_range < 0.0f)
					normalized_rotation_boundaries_range += 2 * M_PIf32;
			}

			void setApplyTranslationBoundary(const bool apply_translation_boundary_) {
				apply_translation_boundary = apply_translation_boundary_;
			}

			void setTranslationBoundary(const float translation_boundary_) {
				translation_boundary = translation_boundary_;
				squared_translation_boundary = translation_boundary * translation_boundary;
			}

			void setTranslationInit(const std::vector<float> translation_init_) {
				translation_init = translation_init_;
			}

			// Return the minimal solver
			const _MinimalSolverEngine *getMinimalSolver() const
			{
				return minimal_solver.get();
			}

			// Return a mutable minimal solver
			_MinimalSolverEngine *getMutableMinimalSolver()
			{
				return minimal_solver.get();
			}

			// Return the minimal solver
			const _NonMinimalSolverEngine *getNonMinimalSolver() const
			{
				return non_minimal_solver.get();
			}

			// Return a mutable minimal solver
			_NonMinimalSolverEngine *getMutableNonMinimalSolver()
			{
				return non_minimal_solver.get();
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			inline size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Estimating the model from a minimal sample
			inline bool estimateModel(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample usd for the estimation
				std::vector<Model>* models_) const // The estimated model parameters
			{
				if (!minimal_solver->estimateModel(data_, // The data points
					sample_, // The sample used for the estimation
					sampleSize(), // The size of a minimal sample
					*models_)) {; // The estimated model parameters
						return false;
					}
				else if (apply_rotation_boundaries || apply_translation_boundary) {
					return isValidToInitTransformModel(models_->back());
				}
				return true;
			}

			// Estimating the model from a non-minimal sample
			inline bool estimateModelNonminimal(const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				const size_t &sample_number_, // The size of a minimal sample
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const // The estimated model parameters
			{
				// Return of there are not enough points for the estimation
				if (sample_number_ < nonMinimalSampleSize())
					return false;

				// The four point fundamental matrix fitting algorithm
				if (!non_minimal_solver->estimateModel(data_,
					sample_,
					sample_number_,
					*models_,
					weights_)) {
					return false;
					}
				else if (apply_rotation_boundaries || apply_translation_boundary) {
					return isValidToInitTransformModel(models_->back());
				}
				return true;
			}

			inline double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			inline double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* s = reinterpret_cast<double *>(point_.data);

				const double &x1 = *s;
				const double &y1 = *(s + 1);
				const double &x2 = *(s + 2);
				const double &y2 = *(s + 3);

				const double t1 = descriptor_(0, 0) * x1 + descriptor_(1, 0) * y1 + descriptor_(2, 0);
				const double t2 = descriptor_(0, 1) * x1 + descriptor_(1, 1) * y1 + descriptor_(2, 1);
				
				const double dx = x2 - t1;
				const double dy = y2 - t2;

				return dx * dx + dy * dy;
			}

			inline double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			inline double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return sqrt(squaredResidual(point_, descriptor_));
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			// TODO NL: Add validation of the sample
			inline bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t *sample_) const // The indices of the selected points
			{
				if (sample_ == nullptr)
					return false;

				// Geometrical checks
				if (check_min_edges_length || check_edges_similarity) {

					const double *data_ptr = reinterpret_cast<double *>(data_.data);
					const int cols = data_.cols;

					size_t offset_i = cols * sample_[0];
					const double
						&x0_i = data_ptr[offset_i],
						&y0_i = data_ptr[offset_i + 1],
						&x1_i = data_ptr[offset_i + 2],
						&y1_i = data_ptr[offset_i + 3];

					size_t offset_j = cols * sample_[1];
					const double
						&x0_j = data_ptr[offset_j],
						&y0_j = data_ptr[offset_j + 1],
						&x1_j = data_ptr[offset_j + 2],
						&y1_j = data_ptr[offset_j + 3];

					double squared_len_edge_source = (x0_i - x0_j) * (x0_i - x0_j) + (y0_i - y0_j) * (y0_i - y0_j);
					double squared_len_edge_target = (x1_i - x1_j) * (x1_i - x1_j) + (y1_i - y1_j) * (y1_i - y1_j);
				
					// Check if the source edges and the target edges are long enough
					if (check_min_edges_length) {
						if (squared_len_edge_source < squared_min_edges_length_threshold ||
						squared_len_edge_target < squared_min_edges_length_threshold) {
						return false;
						}
					}
					// Check if the source edges are similar to the target edges and vice versa
					if (check_edges_similarity) {
						if (squared_len_edge_source < squared_len_edge_target * squared_edges_similarity_threshold ||
						squared_len_edge_target < squared_len_edge_source * squared_edges_similarity_threshold) {
						return false;
						}
					}
				}

				return true;
			}

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			inline bool isValidModel(Model& model,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				// constexpr size_t sample_size = sampleSize();
				const double squared_threshold =
					threshold_ * threshold_;

				// Check the minimal sample if the transformation fits for them well

				/*
				for (size_t sample_idx = 0; sample_idx < 2; ++sample_idx)
				{
				*/

					const size_t &point_idx_0 = minimal_sample_[0];
					const double squared_residual_0 =
						squaredResidual(data_.row(point_idx_0), model);
					if (squared_residual_0 > squared_threshold)
						return false;

					const size_t point_idx_1 = minimal_sample_[1];
					const double squared_residual_1 =
						squaredResidual(data_.row(point_idx_1), model);
					if (squared_residual_1 > squared_threshold)
						return false;

				/*
				}
				*/

				return true;
			}

			inline bool isValidToInitTransformModel(Model&model) const
			{
				if (apply_rotation_boundaries) {
					// double rotation_ = std::atan2(model.descriptor(1,0), model.descriptor(0,0));
					
					// Fast atan2 function
					// https://gist.github.com/volkansalma/2972237
					float x = model.descriptor(0,0);
					float y = model.descriptor(0,1);
					float abs_y = std::fabs(y) + 1e-10f;
					float r = (x - std::copysign(abs_y, x)) / (abs_y + std::fabs(x));
					float angle = M_PI_2f32 - std::copysign(M_PI_4f32, x) + (0.1963f * r * r - 0.9817f) * r;
					float rotation_ = std::copysign(angle, y);

					// https://stackoverflow.com/questions/66799475/how-to-elegantly-find-if-an-angle-is-between-a-range
					float rotation_modulo = fmod(rotation_-rotation_boundaries[0], 2*M_PIf32);
					if (rotation_modulo < 0.0f) {
						rotation_modulo += 2*M_PIf32;
					}

					// fprintf(stderr, "%.3f inside [%.3f, %.3f]?  %.3f, %.3f   ", rotation_, rotation_boundaries[0], rotation_boundaries[1], rotation_modulo, normalized_rotation_boundaries_range);
					
					if (rotation_modulo > normalized_rotation_boundaries_range) {
						// fprintf(stderr, " no\n");
						return false;
					}
					// else {
					//	fprintf(stderr, " yes\n");
					//	return true;
					//}
				}
				
				if (apply_translation_boundary) {
					float translation_diff_x = model.descriptor(2,0)-translation_init[0];
					float translation_diff_y = model.descriptor(2,1)-translation_init[1];
					if (translation_diff_x*translation_diff_x + translation_diff_y*translation_diff_y > squared_translation_boundary) {
						return false;
					}
				}

				return true;



				// test
				// for (float rot = -M_PIf32; rot < M_PIf32; rot += 1) {
				// 	for (float bound_1 = 1.5*(-M_PIf32); bound_1 < 1.5*M_PIf32; bound_1 += 0.5) {
				// 		for (float bound_2 = bound_1 + 0.5; bound_2 < 1.6*M_PIf32; bound_2 += 0.5) {
				// 			fprintf(stderr, "%.3f inside [%.3f, %.3f]?", rot, bound_1, bound_2);
				// 			if (fmod(rot-bound_1, 2*M_PIf32) > normalized_rotation_boundaries_range) {
				// 				fprintf(stderr, " no\n");
				// 				return false;
				// 			}
				// 			else {
				// 				fprintf(stderr, " yes\n");
				// 			}
				// 		}
				// 	}
				// }

				// if (rotation_ - rotation_boundaries[0])%(2*M_PI) > normalized_rotation_boundaries_range {
				// 	fprintf(stderr, "no\n");
				// 	return false;
				// }
				// fprintf(stderr, "yes\n");
				// return true;
			}

		};

		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class RigidTransformationEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			RigidTransformationEstimator() :
				minimal_solver(std::make_shared<_MinimalSolverEngine>()), // Minimal solver engine used for estimating a model from a minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>()) // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			{}
			~RigidTransformationEstimator() {}

			// Return the minimal solver
			const _MinimalSolverEngine *getMinimalSolver() const
			{
				return minimal_solver.get();
			}

			// Return a mutable minimal solver
			_MinimalSolverEngine *getMutableMinimalSolver()
			{
				return minimal_solver.get();
			}

			// Return the minimal solver
			const _NonMinimalSolverEngine *getNonMinimalSolver() const
			{
				return non_minimal_solver.get();
			}

			// Return a mutable minimal solver
			_NonMinimalSolverEngine *getMutableNonMinimalSolver()
			{
				return non_minimal_solver.get();
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			inline size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Estimating the model from a minimal sample
			inline bool estimateModel(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample usd for the estimation
				std::vector<Model>* models_) const // The estimated model parameters
			{
				return minimal_solver->estimateModel(data_, // The data points
					sample_, // The sample used for the estimation
					sampleSize(), // The size of a minimal sample
					*models_); // The estimated model parameters
			}

			// Estimating the model from a non-minimal sample
			inline bool estimateModelNonminimal(const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				const size_t &sample_number_, // The size of a minimal sample
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const // The estimated model parameters
			{
				// Return of there are not enough points for the estimation
				if (sample_number_ < nonMinimalSampleSize())
					return false;

				// The four point fundamental matrix fitting algorithm
				if (!non_minimal_solver->estimateModel(data_,
					sample_,
					sample_number_,
					*models_,
					weights_))
					return false;
				return true;
			}

			inline double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			inline double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* s = reinterpret_cast<double *>(point_.data);

				const double &x1 = *s;
				const double &y1 = *(s + 1);
				const double &z1 = *(s + 2);
				const double &x2 = *(s + 3);
				const double &y2 = *(s + 4);
				const double &z2 = *(s + 5);

				const double t1 = descriptor_(0, 0) * x1 + descriptor_(1, 0) * y1 + descriptor_(2, 0) * z1 + descriptor_(3, 0);
				const double t2 = descriptor_(0, 1) * x1 + descriptor_(1, 1) * y1 + descriptor_(2, 1) * z1 + descriptor_(3, 1);
				const double t3 = descriptor_(0, 2) * x1 + descriptor_(1, 2) * y1 + descriptor_(2, 2) * z1 + descriptor_(3, 2);
				
				const double dx = x2 - t1;
				const double dy = y2 - t2;
				const double dz = z2 - t3;

				return dx * dx + dy * dy + dz * dz;
			}

			inline double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			inline double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return sqrt(squaredResidual(point_, descriptor_));
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			inline bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t *sample_) const // The indices of the selected points
			{
				return true;
			}

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			inline bool isValidModel(Model& model,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				constexpr size_t sample_size = sampleSize();
				const double squared_threshold =
					threshold_ * threshold_;

				// Check the minimal sample if the transformation fits for them well
				for (size_t sample_idx = 0; sample_idx < 3; ++sample_idx)
				{
					const size_t &point_idx = minimal_sample_[sample_idx];
					const double squared_residual =
						squaredResidual(data_.row(point_idx), model);
					if (squared_residual > squared_threshold)
						return false;
				}

				return true;
			}
		};
	}
}