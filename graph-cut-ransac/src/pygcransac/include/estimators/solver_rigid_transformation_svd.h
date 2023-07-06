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

#include <iostream>

#include "solver_engine.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a 2D transformation. A model estimation method and error calculation method are implemented
			class Robust2DRigidTransformationSVDBasedSolver : public SolverEngine
			{
			public:
				Robust2DRigidTransformationSVDBasedSolver()
				{
				}

				~Robust2DRigidTransformationSVDBasedSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of returned solutions
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				inline bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			inline bool Robust2DRigidTransformationSVDBasedSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;
				const double inverse_sample_number_ = 1.0 / sample_number_;
				Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
				Eigen::Vector2d t = Eigen::Vector2d::Zero();
				double scale = 1.0;

				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double centroid0x = 0.0,
					centroid0y = 0.0,
					centroid1x = 0.0,
					centroid1y = 0.0;
				double avg_distance0 = 0.0,
					avg_distance1 = 0.0;
				// FIXME: maybe sqrt(3) instead of sqrt(2)
				static const double sqrt_constant = sqrt(2);

				Eigen::MatrixXd covariance;

				if (sample_ != nullptr && sample_number_ == 2) {

					// Calculate the center of gravity for both point clouds
					size_t offset_i = cols * sample_[0];
					size_t offset_j = cols * sample_[1];
					double weight_i = 1.0;
					double weight_j = 1.0;

					const double
						&x0_i = data_ptr[offset_i],
						&y0_i = data_ptr[offset_i + 1],
						&x1_i = data_ptr[offset_i + 2],
						&y1_i = data_ptr[offset_i + 3],
						&x0_j = data_ptr[offset_j],
						&y0_j = data_ptr[offset_j + 1],
						&x1_j = data_ptr[offset_j + 2],
						&y1_j = data_ptr[offset_j + 3];

					centroid0x = (x0_i + x0_j) / 2,
						centroid0y = (y0_i + y0_j) / 2,
						centroid1x = (x1_i + x1_j) / 2,
						centroid1y = (y1_i + y1_j) / 2;
					
					Eigen::MatrixXd coefficients0(2, 2);
					Eigen::MatrixXd coefficients1(2, 2);

					const double dist0x = x0_i - centroid0x;
					const double dist0y = y0_i - centroid0y;
					const double dist1x = x1_i - centroid1x;
					const double dist1y = y1_i - centroid1y;

					coefficients0(0, 0) = dist0x;
					coefficients0(0, 1) = dist0x;
					coefficients0(1, 0) = dist0y;
					coefficients0(1, 1) = dist0y;

					coefficients1(0, 0) = dist1x;
					coefficients1(0, 1) = dist1x;
					coefficients1(1, 0) = dist1y;
					coefficients1(1, 1) = dist1y;

					avg_distance0 =
						std::sqrt(coefficients0(0, 0) * coefficients0(0, 0) +
							coefficients0(1, 0) * coefficients0(1, 0));

					avg_distance1 =
						std::sqrt(coefficients1(0, 0) * coefficients1(0, 0) +
							coefficients1(1, 0) * coefficients1(1, 0));

					if (weights_ != nullptr) {
						weight_i = weights_[0];
						weight_j = weights_[1];
						coefficients0(0, 0) *= weight_i;
						coefficients0(0, 1) *= weight_j;
						coefficients0(1, 0) *= weight_i;
						coefficients0(1, 1) *= weight_j;
						coefficients1(0, 0) *= weight_i;
						coefficients1(0, 1) *= weight_j;
						coefficients1(1, 0) *= weight_i;
						coefficients1(1, 1) *= weight_j;
					}
					
					const double ratio0 = sqrt_constant / avg_distance0,
						ratio1 = sqrt_constant / avg_distance1;

					coefficients0 *= ratio0;
					coefficients1 *= ratio1;

					covariance = coefficients0 * coefficients1.transpose();

				}
				else {

					double weight = 1.0;
					size_t offset;

					// Calculate the center of gravity for both point clouds
					centroid0x = 0.0,
						centroid0y = 0.0,
						centroid1x = 0.0,
						centroid1y = 0.0;
					
					for (size_t i = 0; i < sample_number_; i++)
					{
						if (sample_ == nullptr)
							offset = cols * i;
						else
							offset = cols * sample_[i];

						const double
							&x0 = data_ptr[offset],
							&y0 = data_ptr[offset + 1],
							&x1 = data_ptr[offset + 2],
							&y1 = data_ptr[offset + 3];

						centroid0x += x0;
						centroid0y += y0;
						centroid1x += x1;
						centroid1y += y1;
					}
					
					centroid0x *= inverse_sample_number_;
					centroid0y *= inverse_sample_number_;
					centroid1x *= inverse_sample_number_;
					centroid1y *= inverse_sample_number_;
					
					Eigen::MatrixXd coefficients0(2, sample_number_);
					Eigen::MatrixXd coefficients1(2, sample_number_);

					avg_distance0 = 0.0;
					avg_distance1 = 0.0;

					for (size_t i = 0; i < sample_number_; i++)
					{
						if (sample_ == nullptr)
						{
							offset = cols * i;
						}
						else
						{
							offset = cols * sample_[i];

						}

						const double
							&x0 = data_ptr[offset],
							&y0 = data_ptr[offset + 1],
							&x1 = data_ptr[offset + 2],
							&y1 = data_ptr[offset + 3];

						coefficients0(0, i) = x0 - centroid0x;
						coefficients0(1, i) = y0 - centroid0y;
						coefficients1(0, i) = x1 - centroid1x;
						coefficients1(1, i) = y1 - centroid1y;

						avg_distance0 +=
							std::sqrt(coefficients0(0, i) * coefficients0(0, i) +
								coefficients0(1, i) * coefficients0(1, i));

						avg_distance1 +=
							std::sqrt(coefficients1(0, i) * coefficients1(0, i) +
								coefficients1(1, i) * coefficients1(1, i));

						if (weights_ != nullptr) {
							weight = weights_[i];
							coefficients0(0, i) *= weight;
							coefficients0(1, i) *= weight;
							coefficients1(0, i) *= weight;
							coefficients1(1, i) *= weight;
						}
						
					}

					avg_distance0 *= inverse_sample_number_;
					avg_distance1 *= inverse_sample_number_;

					const double ratio0 = sqrt_constant / avg_distance0,
						ratio1 = sqrt_constant / avg_distance1;

					coefficients0 *= ratio0;
					coefficients1 *= ratio1;

					covariance = coefficients0 * coefficients1.transpose();
				}

				if (covariance.hasNaN())
					return false;

				// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
				// the solution is linear subspace of dimensionality 2.
				// => use the last two singular std::vectors as a basis of the space
				// (according to SVD properties)
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
					// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
					// to apply SVD to a smaller matrix.
					covariance,
					Eigen::ComputeFullV | Eigen::ComputeFullU);

				R = svd.matrixV() * svd.matrixU().transpose();

				if (R.determinant() < 0)
				{
					Eigen::MatrixXd V = svd.matrixV();
					V.col(1) = -V.col(1);
					R = V * svd.matrixU().transpose();
				}

				scale = avg_distance1 / avg_distance0;

				t(0) = -R(0, 0) * centroid0x - R(0, 1) * centroid0y + centroid1x;
				t(1) = -R(1, 0) * centroid0x - R(1, 1) * centroid0y + centroid1y;
				
				Robust2DRigidTransformation model;
				model.descriptor << R(0, 0), R(1, 0), 0,
					R(0, 1), R(1, 1), 0,
					t(0), t(1), 1;
				models_.push_back(model);
				return true;
			};

			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class RigidTransformationSVDBasedSolver : public SolverEngine
			{
			public:
				RigidTransformationSVDBasedSolver()
				{
				}

				~RigidTransformationSVDBasedSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of returned solutions
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				inline bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			inline bool RigidTransformationSVDBasedSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
				Eigen::Vector3d t = Eigen::Vector3d::Zero();
				double scale = 1.0;

				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double weight = 1.0;
				size_t offset;

				// Calculate the center of gravity for both point clouds
				double centroid0x = 0.0,
					centroid0y = 0.0,
					centroid0z = 0.0,
					centroid1x = 0.0,
					centroid1y = 0.0,
					centroid1z = 0.0;

				for (size_t i = 0; i < sample_number_; i++)
				{
					if (sample_ == nullptr)
						offset = cols * i;
					else
						offset = cols * sample_[i];

					const double
						&x0 = data_ptr[offset],
						&y0 = data_ptr[offset + 1],
						&z0 = data_ptr[offset + 2],
						&x1 = data_ptr[offset + 3],
						&y1 = data_ptr[offset + 4],
						&z1 = data_ptr[offset + 5];

					centroid0x += x0;
					centroid0y += y0;
					centroid0z += z0;
					centroid1x += x1;
					centroid1y += y1;
					centroid1z += z1;
				}
				
				centroid0x /= sample_number_;
				centroid0y /= sample_number_;
				centroid0z /= sample_number_;
				centroid1x /= sample_number_;
				centroid1y /= sample_number_;
				centroid1z /= sample_number_;
				
				Eigen::MatrixXd coefficients0(3, sample_number_);
				Eigen::MatrixXd coefficients1(3, sample_number_);

				double avg_distance0 = 0.0,
					avg_distance1 = 0.0;

				for (size_t i = 0; i < sample_number_; i++)
				{
					if (sample_ == nullptr)
					{
						offset = cols * i;
						if (weights_ != nullptr)
							weight = weights_[i];
					}
					else
					{
						offset = cols * sample_[i];
						if (weights_ != nullptr)
							weight = weights_[i];
					}

					const double
						&x0 = data_ptr[offset],
						&y0 = data_ptr[offset + 1],
						&z0 = data_ptr[offset + 2],
						&x1 = data_ptr[offset + 3],
						&y1 = data_ptr[offset + 4],
						&z1 = data_ptr[offset + 5];

					coefficients0(0, i) = x0 - centroid0x;
					coefficients0(1, i) = y0 - centroid0y;
					coefficients0(2, i) = z0 - centroid0z;
					coefficients1(0, i) = x1 - centroid1x;
					coefficients1(1, i) = y1 - centroid1y;
					coefficients1(2, i) = z1 - centroid1z;

					avg_distance0 +=
						std::sqrt(coefficients0(0, i) * coefficients0(0, i) +
							coefficients0(1, i) * coefficients0(1, i) +
							coefficients0(2, i) * coefficients0(2, i));

					avg_distance1 +=
						std::sqrt(coefficients1(0, i) * coefficients1(0, i) +
							coefficients1(1, i) * coefficients1(1, i) +
							coefficients1(2, i) * coefficients1(2, i));

					coefficients0(0, i) *= weight;
					coefficients0(1, i) *= weight;
					coefficients0(2, i) *= weight;
					coefficients1(0, i) *= weight;
					coefficients1(1, i) *= weight;
					coefficients1(2, i) *= weight;
					 
				}

				avg_distance0 /= sample_number_;
				avg_distance1 /= sample_number_;

				static const double sqrt_3 = std::sqrt(3.0);
				const double ratio0 = sqrt_3 / avg_distance0,
					ratio1 = sqrt_3 / avg_distance1;

				coefficients0 *= ratio0;
				coefficients1 *= ratio1;
				
				Eigen::MatrixXd covariance =
					coefficients0 * coefficients1.transpose();
				
				if (covariance.hasNaN())
					return false;

				// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
				// the solution is linear subspace of dimensionality 2.
				// => use the last two singular std::vectors as a basis of the space
				// (according to SVD properties)
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
					// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
					// to apply SVD to a smaller matrix.
					covariance,
					Eigen::ComputeFullV | Eigen::ComputeFullU);

				R = svd.matrixV() * svd.matrixU().transpose();

				if (R.determinant() < 0)
				{
					//printf("det(R) < R, reflection detected!, correcting for it ...");
					Eigen::MatrixXd V = svd.matrixV();
					V.col(2) = -V.col(2);
					R = V * svd.matrixU().transpose();
				}

				scale = avg_distance1 / avg_distance0;

				t(0) = -R(0, 0) * centroid0x - R(0, 1) * centroid0y - R(0, 2) * centroid0z + centroid1x;
				t(1) = -R(1, 0) * centroid0x - R(1, 1) * centroid0y - R(1, 2) * centroid0z + centroid1y;
				t(2) = -R(2, 0) * centroid0x - R(2, 1) * centroid0y - R(2, 2) * centroid0z + centroid1z;
				
				RigidTransformation model;
				model.descriptor << R(0, 0), R(1, 0), R(2, 0), 0,
					R(0, 1), R(1, 1), R(2, 1), 0,
					R(0, 2), R(1, 2), R(2, 2), 0,
					t(0), t(1), t(2), 1;
				models_.push_back(model);
				return true;
			}
		}
	}
}