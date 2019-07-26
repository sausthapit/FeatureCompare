//
// Created by saurav on 26/07/19.
//

#include <fstream>
#include "QualityEvaluator.h"

string data_path;
const int DATASETS_COUNT = 8;
const int TEST_CASE_COUNT = 5;

const string IMAGE_DATASETS_DIR = "detectors_descriptors_evaluation/images_datasets/";
const string DETECTORS_DIR = "detectors_descriptors_evaluation/detectors/";
const string DESCRIPTORS_DIR = "detectors_descriptors_evaluation/descriptors/";
const string KEYPOINTS_DIR = "detectors_descriptors_evaluation/keypoints_datasets/";

const string PARAMS_POSTFIX = "_params.xml";
const string RES_POSTFIX = "_res.xml";

const string REPEAT = "repeatability";
const string CORRESP_COUNT = "correspondence_count";

string DATASET_NAMES[DATASETS_COUNT] = { "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"};

string DEFAULT_PARAMS = "default";

string IS_ACTIVE_PARAMS = "isActiveParams";
string IS_SAVE_KEYPOINTS = "isSaveKeypoints";
void BaseQualityEvaluator::readAllDatasetsRunParams()
{
    string filename = getRunParamsFilename();
    FileStorage fs( filename, FileStorage::READ );
    if( !fs.isOpened() )
    {
        isWriteParams = true;
        setDefaultAllDatasetsRunParams();
        printf("All runParams are default.\n");
    }
    else
    {
        isWriteParams = false;
        FileNode topfn = fs.getFirstTopLevelNode();

        FileNode pfn = topfn[DEFAULT_PARAMS];
        readDefaultRunParams(pfn);

        for( int i = 0; i < DATASETS_COUNT; i++ )
        {
            FileNode fn = topfn[DATASET_NAMES[i]];
            if( fn.empty() )
            {
                printf( "%d-runParams is default.\n", i);
                setDefaultDatasetRunParams(i);
            }
            else
                readDatasetRunParams(fn, i);
        }
    }
}

void BaseQualityEvaluator::writeAllDatasetsRunParams() const
{
    string filename = getRunParamsFilename();
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "run_params" << "{"; // top file node
        fs << DEFAULT_PARAMS << "{";
        writeDefaultRunParams(fs);
        fs << "}";
        for( int i = 0; i < DATASETS_COUNT; i++ )
        {
            fs << DATASET_NAMES[i] << "{";
            writeDatasetRunParams(fs, i);
            fs << "}";
        }
        fs << "}";
    }
    else
        printf( "File %s for writing run params can not be opened.\n", filename.c_str() );
}

void BaseQualityEvaluator::setDefaultAllDatasetsRunParams()
{
    for( int i = 0; i < DATASETS_COUNT; i++ )
        setDefaultDatasetRunParams(i);
}

bool BaseQualityEvaluator::readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs )
{
    Hs.resize( TEST_CASE_COUNT );
    imgs.resize( TEST_CASE_COUNT+1 );
    string dirname = data_path + IMAGE_DATASETS_DIR + datasetName + "/";
    for( int i = 0; i < (int)Hs.size(); i++ )
    {
        stringstream filename; filename << "H1to" << i+2 << "p.xml";
        FileStorage fs( dirname + filename.str(), FileStorage::READ );
        if( !fs.isOpened() )
        {
            cout << "filename " << dirname + filename.str() << endl;
            FileStorage fs2( dirname + filename.str(), FileStorage::READ );
            return false;
        }
        fs.getFirstTopLevelNode() >> Hs[i];
    }

    for( int i = 0; i < (int)imgs.size(); i++ )
    {
        stringstream filename; filename << "img" << i+1 << ".png";
        imgs[i] = imread( dirname + filename.str(), 0 );
        if( imgs[i].empty() )
        {
            cout << "filename " << filename.str() << endl;
            return false;
        }
    }
    return true;
}

void BaseQualityEvaluator::processResults( int datasetIdx )
{
    if( isWriteGraphicsData )
        writePlotData( datasetIdx );
}

void BaseQualityEvaluator::processResults()
{
    if( isWriteParams )
        writeAllDatasetsRunParams();
}

void BaseQualityEvaluator::run()
{
//    readAlgorithm ();
    processRunParamsFile ();

    int notReadDatasets = 0;
    int progress = 0;

    FileStorage runParamsFS( getRunParamsFilename(), FileStorage::READ );
    isWriteParams = (! runParamsFS.isOpened());
    FileNode topfn = runParamsFS.getFirstTopLevelNode();
    FileNode defaultParams = topfn[DEFAULT_PARAMS];
    readDefaultRunParams (defaultParams);

    cout << testName << endl;
    for(int di = 0; di < DATASETS_COUNT; di++ )
    {
        cout << "Dataset " << di << " [" << DATASET_NAMES[di] << "] " << flush;
        vector<Mat> imgs, Hs;
        if( !readDataset( DATASET_NAMES[di], Hs, imgs ) )
        {
            calcQualityClear (di);
            printf( "Images or homography matrices of dataset named %s can not be read\n",
                    DATASET_NAMES[di].c_str());
            notReadDatasets++;
            continue;
        }

        FileNode fn = topfn[DATASET_NAMES[di]];
        readDatasetRunParams(fn, di);

        runDatasetTest (imgs, Hs, di, progress);
        processResults( di );
        cout << endl;
    }
    if( notReadDatasets == DATASETS_COUNT )
    {
        printf( "All datasets were not be read\n");
        exit(-1);
    }
    else
        processResults();
    runParamsFS.release();
}


string DetectorQualityEvaluator::getRunParamsFilename() const
{
    return data_path + DETECTORS_DIR + algName + PARAMS_POSTFIX;
}

string DetectorQualityEvaluator::getResultsFilename() const
{
    return data_path + DETECTORS_DIR + algName + RES_POSTFIX;
}

string DetectorQualityEvaluator::getPlotPath() const
{
    return data_path + DETECTORS_DIR + "plots/";
}

void DetectorQualityEvaluator::calcQualityClear( int datasetIdx )
{
    calcQuality[datasetIdx].clear();
}

bool DetectorQualityEvaluator::isCalcQualityEmpty( int datasetIdx ) const
{
    return calcQuality[datasetIdx].empty();
}

void DetectorQualityEvaluator::readDefaultRunParams (FileNode &fn)
{
    if (! fn.empty() )
    {
        isSaveKeypointsDefault = (int)fn[IS_SAVE_KEYPOINTS] != 0;
        defaultDetector->read (fn);
    }
}

void DetectorQualityEvaluator::writeDefaultRunParams (FileStorage &fs) const
{
    fs << IS_SAVE_KEYPOINTS << isSaveKeypointsDefault;
    defaultDetector->write (fs);
}

void DetectorQualityEvaluator::readDatasetRunParams( FileNode& fn, int datasetIdx )
{
    isActiveParams[datasetIdx] = (int)fn[IS_ACTIVE_PARAMS] != 0;
    if (isActiveParams[datasetIdx])
    {
        isSaveKeypoints[datasetIdx] = (int)fn[IS_SAVE_KEYPOINTS] != 0;
        specificDetector->read (fn);
    }
    else
    {
        setDefaultDatasetRunParams(datasetIdx);
    }
}

void DetectorQualityEvaluator::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << isActiveParams[datasetIdx];
    fs << IS_SAVE_KEYPOINTS << isSaveKeypoints[datasetIdx];
    defaultDetector->write (fs);
}

void DetectorQualityEvaluator::setDefaultDatasetRunParams( int datasetIdx )
{
    isSaveKeypoints[datasetIdx] = isSaveKeypointsDefault;
    isActiveParams[datasetIdx] = isActiveParamsDefault;
}

void DetectorQualityEvaluator::writePlotData(int di ) const
{
    int imgXVals[] = { 2, 3, 4, 5, 6 }; // if scale, blur or light changes
    int viewpointXVals[] = { 20, 30, 40, 50, 60 }; // if viewpoint changes
    int jpegXVals[] = { 60, 80, 90, 95, 98 }; // if jpeg compression

    int* xVals = 0;
    if( !DATASET_NAMES[di].compare("ubc") )
    {
        xVals = jpegXVals;
    }
    else if( !DATASET_NAMES[di].compare("graf") || !DATASET_NAMES[di].compare("wall") )
    {
        xVals = viewpointXVals;
    }
    else
        xVals = imgXVals;

    stringstream rFilename, cFilename;
    rFilename << getPlotPath() << algName << "_" << DATASET_NAMES[di]  << "_repeatability.csv";
    cFilename << getPlotPath() << algName << "_" << DATASET_NAMES[di]  << "_correspondenceCount.csv";
    ofstream rfile(rFilename.str().c_str()), cfile(cFilename.str().c_str());
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        rfile << xVals[ci] << ", " << calcQuality[di][ci].repeatability << endl;
        cfile << xVals[ci] << ", " << calcQuality[di][ci].correspondenceCount << endl;
    }
}

void DetectorQualityEvaluator::openToWriteKeypointsFile( FileStorage& fs, int datasetIdx )
{
    string filename = data_path + KEYPOINTS_DIR + algName + "_"+ DATASET_NAMES[datasetIdx] + ".xml.gz" ;

    fs.open(filename, FileStorage::WRITE);
    if( !fs.isOpened() )
        printf( "keypoints can not be written in file %s because this file can not be opened\n", filename.c_str() );
}
inline void writeKeypoints( FileStorage& fs, const vector<KeyPoint>& keypoints, int imgIdx )
{
    if( fs.isOpened() )
    {
        stringstream imgName; imgName << "img" << imgIdx;
        write( fs, imgName.str(), keypoints );
    }
}

inline void readKeypoints( FileStorage& fs, vector<KeyPoint>& keypoints, int imgIdx )
{
    assert( fs.isOpened() );
    stringstream imgName; imgName << "img" << imgIdx;
    read( fs[imgName.str()], keypoints);
}

//void DetectorQualityEvaluator::readAlgorithm ()
//{
//    defaultDetector = FeatureDetector::create( algName );
//    specificDetector = FeatureDetector::create( algName );
//    if( defaultDetector == 0 )
//    {
//        printf( "Algorithm can not be read\n" );
//        exit(-1);
//    }
//}

static int update_progress( const string& /*name*/, int progress, int test_case_idx, int count, double dt )
{
    int width = 60 /*- (int)name.length()*/;
    if( count > 0 )
    {
        int t = cvRound( ((double)test_case_idx * width)/count );
        if( t > progress )
        {
            cout << "." << flush;
            progress = t;
        }
    }
    else if( cvRound(dt) > progress )
    {
        cout << "." << flush;
        progress = cvRound(dt);
    }

    return progress;
}
void DetectorQualityEvaluator::runDatasetTest (const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress)
{
    Ptr<FeatureDetector> detector = isActiveParams[di] ? specificDetector : defaultDetector;
    FileStorage keypontsFS;
    if( isSaveKeypoints[di] )
        openToWriteKeypointsFile( keypontsFS, di );

    calcQuality[di].resize(TEST_CASE_COUNT);

    vector<KeyPoint> keypoints1;
    detector->detect( imgs[0], keypoints1 );
    writeKeypoints( keypontsFS, keypoints1, 0);
    int progressCount = DATASETS_COUNT*TEST_CASE_COUNT;

    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        progress = update_progress( testName, progress, di*TEST_CASE_COUNT + ci + 1, progressCount, 0 );
        vector<KeyPoint> keypoints2;
        float rep;
        evaluateFeatureDetector( imgs[0], imgs[ci+1], Hs[ci], &keypoints1, &keypoints2,
                                 rep, calcQuality[di][ci].correspondenceCount,
                                 detector );
        calcQuality[di][ci].repeatability = rep == -1 ? rep : 100.f*rep;
        writeKeypoints( keypontsFS, keypoints2, ci+1);
    }
}