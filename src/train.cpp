/*
 * train.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: Fabian Tschopp
 */

#include "process.hpp"
#include "train.hpp"
#include "filesystem_utils.hpp"
#include "utils.hpp"
#include "caffe/layers/memory_data_layer.hpp"


namespace caffe_neural {

typedef std::map<std::string, int> pmap;

void preload_process_images(TrainImageProcessor& image_processor, InputParam& input_param, pmap extra_param) {
 //unpack params
  unsigned int nr_channels = 0;
  unsigned int nr_labels = 0;
  if (extra_param.count("nr_channels") != 0)
    nr_channels = static_cast<unsigned int>(extra_param.find("nr_channels")->second);
  else
    LOG(INFO) << "Assume number of channels = 0";
  if (extra_param.count("nr_labels") != 0)
    nr_labels = static_cast<unsigned int>(extra_param.find("nr_labels")->second);
  else
    LOG(INFO) << "Assume number of labels = 0";

  if ( input_param.has_preprocessor() )
    image_processor.SetUpParams(input_param, extra_param);

  if(!(input_param.has_raw_images() && input_param.has_label_images())) {
    LOG(FATAL) << "Raw images or label images folder missing.";
  }

  std::set<std::string> filetypes = CreateImageTypesSet();

  int error;
  std::vector<std::vector<bofs::path>> training_set = LoadTrainingSetItems(filetypes, input_param.raw_images(),input_param.label_images(),&error);
  unsigned int ijsum = 0;
  // Preload and preprocess all images
  for (unsigned int i = 0; i < training_set.size(); ++i) {
    std::vector<bofs::path> training_item = training_set[i];

    std::vector<cv::Mat> raw_stack;
    std::vector<std::vector<cv::Mat>> labels_stack(training_item.size() - 1);

    std::string type = bofs::extension(training_item[0]);
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    if(type == ".tif" || type == ".tiff") {
      // TIFF and multipage TIFF mode
      raw_stack = LoadTiff(training_item[0].string(), nr_channels);
    } else {
      // All other image types
      cv::Mat raw_image = cv::imread(training_item[0].string(), nr_channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE :
          CV_LOAD_IMAGE_COLOR);
      raw_stack.push_back(raw_image);
    }
    for(unsigned int k = 0; k < training_item.size() - 1; ++k) {
      std::string type = bofs::extension(training_item[k+1]);
      std::transform(type.begin(), type.end(), type.begin(), ::tolower);
      if(type == ".tif" || type == ".tiff") {
        std::vector<cv::Mat> label_stack = LoadTiff(training_item[k+1].string(), 1);
        labels_stack[k] = label_stack;
      }
      else {
        std::vector<cv::Mat> label_stack;
        cv::Mat label_image = cv::imread(training_item[k+1].string(), CV_LOAD_IMAGE_GRAYSCALE);
        label_stack.push_back(label_image);
        labels_stack[k] = label_stack;
      }
    }
    for (unsigned int j = 0; j < raw_stack.size(); ++j) {
      std::vector<cv::Mat> label_images;
      for(unsigned int k = 0; k < labels_stack.size(); ++k) {
        label_images.push_back(labels_stack[k][j]);
      }

      if(label_images.size() > 1 && nr_labels != 2 && label_images.size() < nr_labels) {
        // Generate complement label
        cv::Mat clabel(label_images[0].rows, label_images[0].cols, CV_8UC(1), 255.0);
        for(unsigned int k = 0; k < label_images.size(); ++k) {
          cv::subtract(clabel,label_images[k],clabel);
        }
        label_images.push_back(clabel);
      }
      image_processor.SubmitImage(raw_stack[j], ijsum, label_images);
      ++ijsum;
    }
  }

  image_processor.Init();

}

int Train(ToolParam &tool_param, CommonSettings &settings) {

  if (tool_param.train_size() <= settings.param_index) {
    LOG(FATAL)<< "Train parameter index does not exist.";
  }

  TrainParam train_param = tool_param.train(settings.param_index);
  InputParam input_param = train_param.input();

  if(!(input_param.has_patch_size() && input_param.has_padding_size() && input_param.has_labels() && input_param.has_channels())) {
    LOG(FATAL) << "Patch size, padding size, label count or channel count parameter missing.";
  }
  int patch_size = input_param.patch_size();
  int padding_size = input_param.padding_size();
  unsigned int nr_labels = input_param.labels();
  unsigned int nr_channels = input_param.channels();

  std::string proto_solver = "";
  if(!train_param.has_solver()) {
    LOG(FATAL) << "Solver prototxt file argument missing";
  }

  proto_solver = train_param.solver();

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(proto_solver, &solver_param);

  int test_interval = solver_param.has_test_interval()?solver_param.test_interval():-1;

  shared_ptr<caffe::Solver<float> >
        solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  if(train_param.has_solverstate()) {
    // Continue from previous solverstate
    const char* solver_state_c = train_param.solverstate().c_str();
    solver->Restore(solver_state_c);
  }

  // Get handles to the test and train network of the Caffe solver
  boost::shared_ptr<caffe::Net<float>> train_net = solver->net();
  boost::shared_ptr<caffe::Net<float>> test_net;
  if(solver->test_nets().size() > 0) {
    test_net = solver->test_nets()[0];
  }

  // Overwrite label count from the desired count to the pre-consolidation count
  if(input_param.has_preprocessor()) {
    PreprocessorParam preprocessor_param = input_param.preprocessor();
    if(preprocessor_param.has_label_consolidate()) {
      nr_labels = preprocessor_param.label_consolidate().label_size();
    }
  }

  TrainImageProcessor image_processor(patch_size, nr_labels);
  TrainImageProcessor test_img_processor(patch_size, nr_labels);

  pmap extra_param;
  extra_param["nr_channels"] = nr_channels;
  extra_param["nr_labels"] = nr_labels;
  extra_param["padding_size"] = padding_size;
  preload_process_images(image_processor, input_param, extra_param);

  if (test_interval != -1) {
    InputParam test_input_param = tool_param.process(settings.param_index).input();
    extra_param["padding_size"] = test_input_param.padding_size();

    preload_process_images(test_img_processor, test_input_param, extra_param);
  } 


  std::vector<long> labelcounter(nr_labels + 1);

  int train_iters = solver_param.has_max_iter()?solver_param.max_iter():0;

  // Do the training
  for (int i = 0; i < train_iters; ++i) {
    std::string debug_string;
    std::vector<cv::Mat> patch;
    std::vector<cv::Mat> images, images_test;
    std::vector<cv::Mat> labels, labels_test;
 
    patch = image_processor.DrawPatchRandom();
    images.push_back(patch[0]);
    labels.push_back(patch[1]);

    //Prepare test images for test stage
    if(test_interval > -1 && i % test_interval == 0) {
      //patch = test_img_processor.DrawPatchRandom();
      InputParam test_input_param = tool_param.process(settings.param_index).input();
      int offset = test_input_param.padding_size();
      int size = test_input_param.patch_size();
      int padding = test_input_param.padding_size();
      cv::Rect img_roi = cv::Rect(offset / 2, offset / 2, size + padding, size + padding);
      cv::Rect label_roi = cv::Rect(offset / 2, offset / 2, size, size);
      images_test.push_back(cv::Mat(test_img_processor.raw_images()[0], img_roi).clone());
      labels_test.push_back(cv::Mat(test_img_processor.label_images()[0], label_roi).clone());
    }
    
//    std::cout << "image.size(): " << patch[0].size() << (patch.size() > 2)?(std::cout << " vs. " << patch[2].size() << std::endl):(std::cout << std::endl);
//    std::cout << "lables.size(): " << patch[1].size() << (patch.size() > 2)?(std::cout << " vs. " << patch[3].size() << std::endl):(std::cout << std::endl);

    

    
    // TODO: Only enable in debug or statistics mode
    for (int y = 0; y < patch_size; ++y) {
      for (int x = 0; x < patch_size; ++x) {
        labelcounter[patch[1].at<float>(y, x) + 1] += 1;
      }
    }

    if(settings.debug) {
      for (unsigned int k = 0; k < nr_labels + 1; ++k) {
        std::cout << "Label: " << ((int)k - 1) << ", " << labelcounter[k] << std::endl;
      }
    }

    if(settings.graphic) {

      cv::Mat test;

      double minVal, maxVal;
      cv::minMaxLoc(patch[1], &minVal, &maxVal);
      patch[1].convertTo(test, CV_32FC1, 1.0 / (maxVal - minVal),
          -minVal * 1.0 / (maxVal - minVal));

      std::vector<cv::Mat> tv;
      tv.push_back(test);
      tv.push_back(test);
      tv.push_back(test);
      cv::Mat tvl;

      cv::merge(tv, tvl);

      cv::Mat patchclone = patch[0].clone();

      tvl.copyTo(
          patchclone(
              cv::Rect(padding_size / 2, padding_size / 2, patch_size,
                  patch_size)));

      cv::imshow(OCVDBGW, patchclone);
      cv::waitKey(10);
    }
    
    // The labels
    std::vector<int_tp> lalabels;
    lalabels.push_back(0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
        train_net->layers()[0])->AddMatVector(labels, lalabels);

    // The images
    std::vector<int_tp> imlabels;
    imlabels.push_back(0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
        train_net->layers()[1])->AddMatVector(images, imlabels);

    if(test_interval > -1 && i % test_interval == 0) {
      // The labels
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          test_net->layers()[0])->AddMatVector(labels_test, lalabels);

      // The images
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          test_net->layers()[1])->AddMatVector(images_test, imlabels);
    }

    solver->Step(1L);
    if (train_param.has_filter_output()) {
      FilterOutputParam filter_param = train_param.filter_output();
      if (filter_param.has_output_filters() && filter_param.output_filters() && filter_param.has_output()) {
        ExportFilters(solver->net().get(), filter_param.output(), bofs::path("train"), 0, 0, 0, true);
      }
    }

    if(test_interval > -1 && i % test_interval == 0) {
      // TODO: Run tests with the testset and testnet
      // TODO: Apply ISBI and other quality measures (cross, rand, pixel, warp, loss)
      // TODO: Write out statistics to file
    }
  }

  LOG(INFO) << "Training done!";

  return 0;
}
}  // namespace caffe_neural
