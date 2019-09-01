//
// Created by saurav on 26/07/19.
//

#include <fstream>
#include "QualityEvaluator.h"

string data_path="/media/saurav/Data/Datasets/oxford_affine/";
const int DATASETS_COUNT = 1;
const int TEST_CASE_COUNT = 5;

const string IMAGE_DATASETS_DIR = "inputs/";
const string DETECTORS_DIR = "outputs/detectors/";
const string DESCRIPTORS_DIR = "outputs/descriptors/";
const string KEYPOINTS_DIR = "outputs/keypoints_datasets/";

const string PARAMS_POSTFIX = "_params.xml";
const string RES_POSTFIX = "_res.xml";

const string REPEAT = "repeatability";
const string CORRESP_COUNT = "correspondence_count";

//string DATASET_NAMES[DATASETS_COUNT] = { "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"};
string DATASET_NAMES[DATASETS_COUNT] = { "bikes"};

string DEFAULT_PARAMS = "default";

string IS_ACTIVE_PARAMS = "isActiveParams";
string IS_SAVE_KEYPOINTS = "isSaveKeypoints";

/****************************************************************************************\
*                                  Descriptors evaluation                                 *
\****************************************************************************************/

const string RECALL = "recall";
const string PRECISION = "precision";

const string KEYPOINTS_FILENAME = "keypointsFilename";
const string PROJECT_KEYPOINTS_FROM_1IMAGE = "projectKeypointsFrom1Image";
const string MATCH_FILTER = "matchFilter";
const string RUN_PARAMS_IS_IDENTICAL = "runParamsIsIdentical";

const string ONE_WAY_TRAIN_DIR = "detectors_descriptors_evaluation/one_way_train_images/";
const string ONE_WAY_IMAGES_LIST = "one_way_train_images.txt";
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
        stringstream filename; filename << "img" << i+1 << ".ppm";
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

string DescriptorQualityEvaluator::getRunParamsFilename() const
{
    return data_path + DESCRIPTORS_DIR + algName + PARAMS_POSTFIX;
}

string DescriptorQualityEvaluator::getResultsFilename() const
{
    return data_path + DESCRIPTORS_DIR + algName + RES_POSTFIX;
}

string DescriptorQualityEvaluator::getPlotPath() const
{
    return data_path + DESCRIPTORS_DIR + "plots/";
}

void DescriptorQualityEvaluator::calcQualityClear( int datasetIdx )
{
    calcQuality[datasetIdx].clear();
}

bool DescriptorQualityEvaluator::isCalcQualityEmpty( int datasetIdx ) const
{
    return calcQuality[datasetIdx].empty();
}

void DescriptorQualityEvaluator::readDefaultRunParams (FileNode &fn)
{
    if (! fn.empty() )
    {
        commRunParamsDefault.projectKeypointsFrom1Image = (int)fn[PROJECT_KEYPOINTS_FROM_1IMAGE] != 0;
        commRunParamsDefault.matchFilter = (int)fn[MATCH_FILTER];
        defaultDescMatcher->read (fn);
    }
}

void DescriptorQualityEvaluator::writeDefaultRunParams (FileStorage &fs) const
{
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParamsDefault.projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParamsDefault.matchFilter;
    defaultDescMatcher->write (fs);
}

void DescriptorQualityEvaluator::readDatasetRunParams( FileNode& fn, int datasetIdx )
{
    commRunParams[datasetIdx].isActiveParams = (int)fn[IS_ACTIVE_PARAMS] != 0;
    if (commRunParams[datasetIdx].isActiveParams)
    {
        commRunParams[datasetIdx].keypontsFilename = (string)fn[KEYPOINTS_FILENAME];
        commRunParams[datasetIdx].projectKeypointsFrom1Image = (int)fn[PROJECT_KEYPOINTS_FROM_1IMAGE] != 0;
        commRunParams[datasetIdx].matchFilter = (int)fn[MATCH_FILTER];
        specificDescMatcher->read (fn);
    }
    else
    {
        setDefaultDatasetRunParams(datasetIdx);
    }
}

void DescriptorQualityEvaluator::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << commRunParams[datasetIdx].isActiveParams;
    fs << KEYPOINTS_FILENAME << commRunParams[datasetIdx].keypontsFilename;
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParams[datasetIdx].projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParams[datasetIdx].matchFilter;

    defaultDescMatcher->write (fs);
}

void DescriptorQualityEvaluator::setDefaultDatasetRunParams( int datasetIdx )
{
    commRunParams[datasetIdx] = commRunParamsDefault;
    commRunParams[datasetIdx].keypontsFilename = "SURF_" + DATASET_NAMES[datasetIdx] + ".xml.gz";
}

void DescriptorQualityEvaluator::writePlotData( int di ) const
{
    stringstream filename;
    filename << getPlotPath() << algName << "_" << DATASET_NAMES[di] << ".csv";
    FILE *file = fopen (filename.str().c_str(), "w");
    size_t size = calcDatasetQuality[di].size();
    for (size_t i=0;i<size;i++)
    {
        fprintf( file, "%f, %f\n", 1 - calcDatasetQuality[di][i].precision, calcDatasetQuality[di][i].recall);
    }
    fclose( file );
}

void DescriptorQualityEvaluator::readAlgorithm( )
{
    defaultDescMatcher = DescriptorMatcher::create( algName );
    specificDescMatcher = DescriptorMatcher::create( algName );

    if( defaultDescMatcher == 0 )
    {
        Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( algName );
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( matcherName );
        defaultDescMatcher = new VectorDescriptorMatch( extractor, matcher );
        specificDescMatcher = new VectorDescriptorMatch( extractor, matcher );

        if( extractor == 0 || matcher == 0 )
        {
            printf("Algorithm can not be read\n");
            exit(-1);
        }
    }
}

void DescriptorQualityEvaluator::calculatePlotData( vector<vector<DMatch> > &allMatches, vector<vector<uchar> > &allCorrectMatchesMask, int di )
{
    vector<Point2f> recallPrecisionCurve;
    computeRecallPrecisionCurve( allMatches, allCorrectMatchesMask, recallPrecisionCurve );

    calcDatasetQuality[di].clear();
    const float resultPrecision = 0.5;
    bool isResultCalculated = false;
    const double eps = 1e-2;

    Quality initQuality;
    initQuality.recall = 0;
    initQuality.precision = 0;
    calcDatasetQuality[di].push_back( initQuality );

    for( size_t i=0;i<recallPrecisionCurve.size();i++ )
    {
        Quality quality;
        quality.recall = recallPrecisionCurve[i].y;
        quality.precision = 1 - recallPrecisionCurve[i].x;
        Quality back = calcDatasetQuality[di].back();

        if( fabs( quality.recall - back.recall ) < eps && fabs( quality.precision - back.precision ) < eps )
            continue;

        calcDatasetQuality[di].push_back( quality );

        if( !isResultCalculated && quality.precision < resultPrecision )
        {
            for(int ci=0;ci<TEST_CASE_COUNT;ci++)
            {
                calcQuality[di][ci].recall = quality.recall;
                calcQuality[di][ci].precision = quality.precision;
            }
            isResultCalculated = true;
        }
    }
}

void DescriptorQualityEvaluator::runDatasetTest (const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress)
{
    FileStorage keypontsFS( data_path + KEYPOINTS_DIR + commRunParams[di].keypontsFilename, FileStorage::READ );
    if( !keypontsFS.isOpened())
    {
        calcQuality[di].clear();
        printf( "keypoints from file %s can not be read\n", commRunParams[di].keypontsFilename.c_str() );
        return;
    }

    Ptr<GenericDescriptorMatcher> descMatch = commRunParams[di].isActiveParams ? specificDescMatcher : defaultDescMatcher;
    calcQuality[di].resize(TEST_CASE_COUNT);

    vector<KeyPoint> keypoints1;
    readKeypoints( keypontsFS, keypoints1, 0);

    int progressCount = DATASETS_COUNT*TEST_CASE_COUNT;

    vector<vector<DMatch> > allMatches1to2;
    vector<vector<uchar> > allCorrectMatchesMask;
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        progress = update_progress( testName, progress, di*TEST_CASE_COUNT + ci + 1, progressCount, 0 );

        vector<KeyPoint> keypoints2;
        if( commRunParams[di].projectKeypointsFrom1Image )
        {
            // TODO need to test function calcKeyPointProjections
            calcKeyPointProjections( keypoints1, Hs[ci], keypoints2 );
            filterKeyPointsByImageSize( keypoints2,  imgs[ci+1].size() );
        }
        else
            readKeypoints( keypontsFS, keypoints2, ci+1 );
        // TODO if( commRunParams[di].matchFilter )

        vector<vector<DMatch> > matches1to2;
        vector<vector<uchar> > correctMatchesMask;
        vector<Point2f> recallPrecisionCurve; // not used because we need recallPrecisionCurve for
        // all images in dataset
        evaluateGenericDescriptorMatcher( imgs[0], imgs[ci+1], Hs[ci], keypoints1, keypoints2,
                                          &matches1to2, &correctMatchesMask, recallPrecisionCurve,
                                          descMatch );
        allMatches1to2.insert( allMatches1to2.end(), matches1to2.begin(), matches1to2.end() );
        allCorrectMatchesMask.insert( allCorrectMatchesMask.end(), correctMatchesMask.begin(), correctMatchesMask.end() );
    }

    calculatePlotData( allMatches1to2, allCorrectMatchesMask, di );
}

//--------------------------------- Calonder descriptor test --------------------------------------------
class CalonderDescriptorQualityEvaluator : public DescriptorQualityEvaluator
{
public:
    CalonderDescriptorQualityEvaluator() :
            DescriptorQualityEvaluator( "Calonder", "quality-descriptor-calonder") {}
    virtual void readAlgorithm( )
    {
        string classifierFile = data_path + "/features2d/calonder_classifier.rtc";
        defaultDescMatcher = new VectorDescriptorMatch( new CalonderDescriptorExtractor<float>( classifierFile ),
                                                        new BFMatcher(NORM_L2) );
        specificDescMatcher = defaultDescMatcher;
    }
};