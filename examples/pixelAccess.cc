#include "Eigen/Core"
#include <memory>
#include "boost/timer.hpp"

#include "lsst/afw/image.h"
#include "lsst/geom.h"
#include "lsst/ip/diffim/ImageSubtract.h"

namespace geom = lsst::geom;
namespace afwImage = lsst::afw::image;
namespace diffim = lsst::ip::diffim;

template <typename ImageT>
Eigen::MatrixXd test(afwImage::Image<ImageT> varianceEstimate,
                     int cswitch)
{

    /*
       Each entry in M is the sum of :
       all the pixels in 3 images multiplied together
       the pixels are at the same x,y position in each image
       Two of the images are stored in imageList
       and you multiply them by each other i*j with i = 0..N; j = i..N
       so only the upper triangular half of M
       The third image is in the denomiator, e.g. variance
    */

    unsigned int const nParameters = 400;
    Eigen::MatrixXd mMat = Eigen::MatrixXd::Zero(nParameters, nParameters);

    /* iterate over a subset of the pixels in each image */
    int const startCol = 5;
    int const startRow = 5;
    int const endCol   = varianceEstimate.getWidth() - startCol;
    int const endRow   = varianceEstimate.getHeight() - startRow;

    if (cswitch == 3) {
        /* a list of images - in diffim each one of these is associated with a basis function */
        std::vector<Eigen::VectorXd> imageList(nParameters);
        typename std::vector<Eigen::VectorXd>::iterator eiter = imageList.begin();
        afwImage::Image<ImageT> cimage(varianceEstimate.getDimensions());
        for (int i = 1; eiter != imageList.end(); ++eiter, ++i) {
            cimage = i; /* give it a value */
            Eigen::MatrixXd cMat = diffim::imageToEigenMatrix(cimage).block(startRow, startCol,
                                                                            endRow-startRow,
                                                                            endCol-startCol);
            cMat.resize(cMat.rows()*cMat.cols(), 1);
            Eigen::VectorXd vMat = cMat.col(0);
            *eiter = vMat;
        }

        /* eigen representation of input images; only the pixels that are unconvolved in cimage below */
        Eigen::MatrixXd eigeniVarianceM =
            diffim::imageToEigenMatrix(varianceEstimate).block(startRow, startCol,
                                                               endRow-startRow,
                                                               endCol-startCol).array().inverse().matrix();
        eigeniVarianceM.resize(eigeniVarianceM.rows()*eigeniVarianceM.cols(), 1);
        Eigen::VectorXd eigeniVarianceV      = eigeniVarianceM.col(0);

        Eigen::MatrixXd cMat(eigeniVarianceV.size(), nParameters);
        typename std::vector<Eigen::VectorXd>::iterator eiterj = imageList.begin();
        typename std::vector<Eigen::VectorXd>::iterator eiterE = imageList.end();
        for (unsigned int kidxj = 0; eiterj != eiterE; eiterj++, kidxj++) {
            cMat.block(0, kidxj, eigeniVarianceV.size(), 1) =
                Eigen::MatrixXd(*eiterj).block(0, 0, eigeniVarianceV.size(), 1);
        }

        // Caculate the variance-weighted pixel values
        Eigen::MatrixXd vcMat = eigeniVarianceV.asDiagonal() * cMat;

        // Calculate M as the variance-weighted inner product of C
        //M.part<Eigen::SelfAdjoint>() = (C.transpose() * VC).lazy();
        mMat = cMat.transpose() * vcMat;
        return mMat;
    }
    else if (cswitch == 2) {
        /* a list of images - in diffim each one of these is associated with a basis function */
        std::vector<Eigen::VectorXd> imageList(nParameters);
        typename std::vector<Eigen::VectorXd>::iterator eiter = imageList.begin();
        afwImage::Image<ImageT> cimage(varianceEstimate.getDimensions());
        for (int i = 1; eiter != imageList.end(); ++eiter, ++i) {
            cimage = i; /* give it a value */
            Eigen::MatrixXd cMat = diffim::imageToEigenMatrix(cimage).block(startRow,
                                                                            startCol,
                                                                            endRow-startRow,
                                                                            endCol-startCol);
            cMat.resize(cMat.rows()*cMat.cols(), 1);
            Eigen::VectorXd vMat = cMat.col(0);
            *eiter = vMat;
        }

        /* eigen representation of input images; only the pixels that are unconvolved in cimage below */
        Eigen::MatrixXd eigeniVarianceM =
            diffim::imageToEigenMatrix(varianceEstimate).block(startRow, startCol,
                                                               endRow-startRow,
                                                               endCol-startCol).array().inverse().matrix();
        eigeniVarianceM.resize(eigeniVarianceM.rows()*eigeniVarianceM.cols(), 1);
        Eigen::VectorXd eigeniVarianceV      = eigeniVarianceM.col(0);

        typename std::vector<Eigen::VectorXd>::iterator eiteri = imageList.begin();
        typename std::vector<Eigen::VectorXd>::iterator eiterE = imageList.end();
        for (unsigned int kidxi = 0; eiteri != eiterE; eiteri++, kidxi++) {
            Eigen::VectorXd eiteriDotiVariance = (eiteri->array() * eigeniVarianceV.array()).matrix();

            typename std::vector<Eigen::VectorXd>::iterator eiterj = eiteri;
            for (unsigned int kidxj = kidxi; eiterj != eiterE; eiterj++, kidxj++) {
                mMat(kidxi, kidxj) = (eiteriDotiVariance.array() * eiterj->array()).sum();
                mMat(kidxj, kidxi) = mMat(kidxi, kidxj);
            }
        }
        return mMat;
    }
    else {

        /* a list of images - in diffim each one of these is associated with a basis function */
        std::vector<std::shared_ptr<afwImage::Image<ImageT> > > imageList(nParameters);
        typename std::vector<std::shared_ptr<afwImage::Image<ImageT> > >::iterator citer=imageList.begin();
        for (int i = 1; citer != imageList.end(); ++citer, ++i) {
            *citer = std::shared_ptr<afwImage::Image<ImageT>>(
                new afwImage::Image<ImageT>(varianceEstimate.getDimensions())
                );
            **citer = i; /* give it a value */
        }

        /* pixel locators */
        std::vector<typename afwImage::Image<ImageT>::xy_locator> locatorList;
        for (citer = imageList.begin(); citer != imageList.end(); ++citer) {
            locatorList.push_back( (**citer).xy_at(startCol,startRow) );
        }
        typename afwImage::Image<ImageT>::xy_locator varianceLocator =
            varianceEstimate.xy_at(startCol, startRow);

        /* at end of each row, this steps in column back to starting col pixel of next row */
        std::pair<int, int> rowStep = std::make_pair(static_cast<int>(-(endCol-startCol)), 1);

        /* now step over the pixels explicitly */
        for (int row = startRow; row < endRow; ++row) {
            for (int col = startCol; col < endCol; ++col) {
                double const iVariance        = 1.0 / *varianceLocator;

                typename std::vector<typename afwImage::Image<ImageT>::xy_locator>::iterator citeri =
                    locatorList.begin();
                typename std::vector<typename afwImage::Image<ImageT>::xy_locator>::iterator citerE =
                    locatorList.end();
                for (int kidxi = 0; citeri != citerE; ++citeri, ++kidxi) {
                    ImageT const cdImagei = **citeri * iVariance;

                    typename std::vector<typename afwImage::Image<ImageT>::xy_locator>::iterator citerj =
                        citeri;
                    for (int kidxj = kidxi; citerj != citerE; ++citerj, ++kidxj) {
                        mMat(kidxi, kidxj) += cdImagei * **citerj;
                    }
                }

                // Step each accessor in column
                ++varianceLocator.x();
                for (unsigned int ki = 0; ki < nParameters; ++ki) {
                    ++locatorList[ki].x();
                }

            } // col
            // Get to next row, first col
            varianceLocator           += rowStep;
            for (unsigned int ki = 0; ki < nParameters; ++ki) {
                locatorList[ki] += rowStep;
            }
        } // row
        // Fill in rest of M
        for (unsigned int kidxi=0; kidxi < nParameters; ++kidxi) {
            for (unsigned int kidxj=kidxi+1; kidxj < nParameters; ++kidxj) {
                mMat(kidxj, kidxi) = mMat(kidxi, kidxj);
            }
        }

        return mMat;
    }
}

int main() {
    boost::timer t;

    afwImage::Image<float> varianceEstimate(geom::Extent2I(100, 100));
    varianceEstimate = 1;

    t.restart();
    Eigen::MatrixXd m1 = ::test(varianceEstimate, 1);
    double time = t.elapsed();
    std::cout << "Manual pixel iteration = " << time << std::endl;

    t.restart();
    Eigen::MatrixXd m2 = ::test(varianceEstimate, 2);
    time = t.elapsed();
    std::cout << "Eigen pixel iteration = " << time << std::endl;

    t.restart();
    Eigen::MatrixXd m3 = ::test(varianceEstimate, 3);
    time = t.elapsed();
    std::cout << "Eigen2 pixel iteration = " << time << std::endl;

    std::cout << (m1-m2).sum() << std::endl;
    std::cout << (m1-m3).sum() << std::endl;

    /*
       On 2.4 GHz Intel Core 2 Duo running Mac OS X 10.5.8 and using gcc 4.0.1

       100 parameters
       Using no opt:
       Manual pixel iteration = 3.7423
       Eigen pixel iteration = 3.25677

       Using opt=3:
       Manual pixel iteration = 0.503348
       Eigen pixel iteration = 0.181815

       200 parameters
       Using no opt:
       Manual pixel iteration = 15.3065
       Eigen pixel iteration = 12.5825

       Using opt=3:
       Manual pixel iteration = 2.18445
       Eigen pixel iteration = 0.576351

       400 parameters
       Using no opt:
       Manual pixel iteration = 67.4297
       Eigen pixel iteration = 49.2621

       Using opt=3:
       Manual pixel iteration = 11.4916
       Eigen pixel iteration = 2.10081

       On quad Intel(R) Xeon(TM) CPU 2.80GHz running RHEL5 and g++ 4.1.2
       400 parameters, opt=3
       Manual pixel iteration = 16.82
       Eigen pixel iteration = 2.4




       NOTE : Took one of the multiplications out of the inner loop to
       speed things up.  This improved the Eigen timing significantly,
       but not so much the afw loop.  Suggesting its the iterators
       accounting for the runtime.

       Manual pixel iteration = 16.65
       Eigen pixel iteration = 1.86




       FINAL NOTE : With help from Mike Jarvis, I was able to speed
       things up 5-10% more.

       Manual pixel iteration = 17.8
       Eigen pixel iteration = 1.86
       Eigen2 pixel iteration = 1.76

    */
}
