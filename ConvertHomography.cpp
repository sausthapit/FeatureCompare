//
// Created by saurav on 30/08/19.
//


#include <iostream>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <fstream>

namespace fs=std::experimental::filesystem;

void convertOneFile(fs::path fname){
    double a,b,c;
    cv::Mat H;
    std::ifstream infile(fname);
    fs::path outfile=fname;
    outfile+=".xml";
    cv::FileStorage storage(fs::absolute(outfile),cv::FileStorage::WRITE);
    int row=0;
    while (infile >>a>>b>>c){
        H.push_back(a);
        H.push_back(b);
        H.push_back(c);
        row++;
    }
    H=H.reshape(1,row);
    storage<<"H"<<H;
    storage.release();
    infile.close();

}

int main(int argc, char** args){

//    fs::path directory("/media/saurav/Data/Datasets/oxford_affine/bikes/");
    fs::path directory(args[1]);
    if(fs::exists(directory)&&fs::is_directory(directory)){
        for (const auto& entry : fs::directory_iterator(directory)){
            auto filename=entry.path().filename();
            if(0==filename.generic_string().find("H1to"))
            {
//                Dont convert xml or yml files again
                int a=filename.generic_string().find(".xml");
                int b=filename.generic_string().find(".yml");
                if(a>0 ||b>0)
                    continue;
//                std::cout<<filename<<std::endl;
                convertOneFile(entry);
            }

        }
    }

    return 0;
}
