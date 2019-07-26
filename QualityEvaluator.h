//
// Created by saurav on 26/07/19.
//

#ifndef FEATURECOMPARE_QUALITYEVALUATOR_H
#define FEATURECOMPARE_QUALITYEVALUATOR_H

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
using namespace std;
using namespace cv;

extern const int DATASETS_COUNT;
class BaseQualityEvaluator
{
public:
    BaseQualityEvaluator(const char* _algName, const char* _testName ) : algName(_algName), testName(_testName)
    {
        //TODO: change this
        isWriteGraphicsData = true;
    }

    void run();

    virtual ~BaseQualityEvaluator(){}

protected:
    virtual string getRunParamsFilename() const = 0;
    virtual string getResultsFilename() const = 0;
    virtual string getPlotPath() const = 0;

    virtual void calcQualityClear( int datasetIdx ) = 0;
    virtual bool isCalcQualityEmpty( int datasetIdx ) const = 0;

    void readAllDatasetsRunParams();
    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx ) = 0;
    void writeAllDatasetsRunParams() const;
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const = 0;
    void setDefaultAllDatasetsRunParams();
    virtual void setDefaultDatasetRunParams( int datasetIdx ) = 0;
    virtual void readDefaultRunParams( FileNode& /*fn*/ ) {}
    virtual void writeDefaultRunParams( FileStorage& /*fs*/ ) const {}

    bool readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs );

    virtual void readAlgorithm() {}
    virtual void processRunParamsFile() {}
    virtual void runDatasetTest( const vector<Mat>& /*imgs*/, const vector<Mat>& /*Hs*/, int /*di*/, int& /*progress*/ ) {}

    virtual void processResults( int datasetIdx );
    virtual void processResults();
    virtual void writePlotData( int /*datasetIdx*/ ) const {}

    string algName, testName;
    bool isWriteParams, isWriteGraphicsData;
};

class DetectorQualityEvaluator : public BaseQualityEvaluator
{
public:
    DetectorQualityEvaluator( Ptr<Feature2D> algo, const char* _detectorName, const char* _testName ) : BaseQualityEvaluator( _detectorName, _testName )
    {
        this->specificDetector=algo;
        this->defaultDetector=algo;
        calcQuality.resize(DATASETS_COUNT);
        isSaveKeypoints.resize(DATASETS_COUNT);
        isActiveParams.resize(DATASETS_COUNT);

        isSaveKeypointsDefault = false;
        isActiveParamsDefault = false;
    }

protected:
    virtual string getRunParamsFilename() const;
    virtual string getResultsFilename() const;
    virtual string getPlotPath() const;

    virtual void calcQualityClear( int datasetIdx );
    virtual bool isCalcQualityEmpty( int datasetIdx ) const;

    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx );
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const;
    virtual void setDefaultDatasetRunParams( int datasetIdx );
    virtual void readDefaultRunParams( FileNode &fn );
    virtual void writeDefaultRunParams( FileStorage &fs ) const;

    virtual void writePlotData( int di ) const;

    void openToWriteKeypointsFile( FileStorage& fs, int datasetIdx );

//    virtual void readAlgorithm();
    virtual void processRunParamsFile() {}
    virtual void runDatasetTest( const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress );

    Ptr<FeatureDetector> specificDetector;
    Ptr<FeatureDetector> defaultDetector;

    struct Quality
    {
        float repeatability;
        int correspondenceCount;
    };
    vector<vector<Quality> > calcQuality;

    vector<bool> isSaveKeypoints;
    vector<bool> isActiveParams;

    bool isSaveKeypointsDefault;
    bool isActiveParamsDefault;
};
#endif //FEATURECOMPARE_QUALITYEVALUATOR_H
