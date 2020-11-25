// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to use dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014

    In particular, we will train a face landmarking model based on a small dataset
    and then evaluate it.  If you want to visualize the output of the trained
    model on some images then you can run the face_landmark_detection_ex.cpp
    example program with sp.dat as the input model.

    It should also be noted that this kind of model, while often used for face
    landmarking, is quite general and can be used for a variety of shape
    prediction tasks.  But here we demonstrate it only on a simple face
    landmarking task.
*/


#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/cmd_line_parser/get_option.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances(
  const std::vector<std::vector<full_object_detection> >& objects
);
/*!
    ensures
        - returns an object D such that:
            - D[i][j] == the distance, in pixels, between the eyes for the face represented
              by objects[i][j].
!*/

// ----------------------------------------------------------------------------------------

void writeConfigToFile(std::string configFile, shape_predictor_trainer trainer) {
  ofstream file;
  file.open(configFile);
  file << "Landmarks trainer configuration used\n";
  file << "---------------------------------------------------------------------------\n";
  file << "Cascade depth: " << trainer.get_cascade_depth() << "\n";
  file << "Feature pool region padding: " << trainer.get_feature_pool_region_padding() << "\n";
  file << "Feature pool size: " << trainer.get_feature_pool_size() << "\n";
  file << "Lambda: " << trainer.get_lambda() << "\n";
  file << "Nu: " << trainer.get_nu() << "\n";
  file << "Num test splits: " << trainer.get_num_test_splits() << "\n";
  file << "Num threads: " << trainer.get_num_threads() << "\n";
  file << "Num trees per cascade level: " << trainer.get_num_trees_per_cascade_level() << "\n";
  file << "Oversampling amount: " << trainer.get_oversampling_amount() << "\n";
  file << "Oversampling translation jitter: " << trainer.get_oversampling_translation_jitter() << "\n";
  file << "Padding mode: " << trainer.get_padding_mode() << "\n";
  file << "Random seed: " << trainer.get_random_seed() << "\n";
  file << "Tree depth: " << trainer.get_tree_depth() << "\n";
  file.close();
}

int main(int argc, char** argv)
{
  try
  {
    command_line_parser parser;
    parser.add_option("dataset", "Directory containing the dataset.", 1);
    parser.add_option("td", "Depth used in trees generated for training. More depth more accuracy in preditions (also model size increase).", 1);
    parser.add_option("threads", "Number of threads used in training.", 1);
    parser.add_option("oversampling", "Number of oversampling amount needed", 1);
    parser.add_option("cascade", "Cascade Depth is the number of cascades used to train the model. This parameter affect either the size and accuracy of a model.", 1);
    parser.add_option("nu", "Nu is the regularization parameter. It determines the ability of the model to generalize and learn patterns instead of fixed-data.", 1);
    parser.add_option("test-splits", "Is the number of split features sampled at each node. This parameter is responsible for selecting the best features at each cascade during the training process. The parameter affects the training speed and the model accuracy.", 1);
    parser.add_option("features-pool-size", "Feature Pool Size denotes the number of pixels used to generate the features for the random trees at each cascade. Larger amount of pixels will lead the algorithm to be more robust and accurate but to execute slower.", 1);
    parser.add_option("save-to", "Save model to fillename expecified.", 1);
    parser.add_option("h", "Display this help message.");

    // now I will parse the command line
    parser.parse(argc, argv);

    // check if the -h option was given on the command line
    if (parser.option("h") || argc < 2)
    {
      // display all the command line options
      cout << "Usage: " << argv[0] << " --dataset path_to_dataset  --td (Trees depht: default 2) --threads (Number of threads for training: default 2)\n";
      // This function prints out a nicely formatted list of
      // all the options the parser has
      parser.print_options();
      return 0;
    }

    std::string dataset_directory;
    if (parser.option("dataset")) {
      dataset_directory = parser.option("dataset").argument();
    }
    else {
      cout << "Error in arguments: You must supply dataset location path." << endl;
      return -1;
    }

    std::string modelFilename;
    if (parser.option("save-to")) {
      modelFilename = parser.option("save-to").argument();
    }
    else {
      modelFilename = "sp.dat";
    }

    // We obtain the params or its default values
    int treeDepthParam = get_option(parser, "td", 2);
    int threadsParam = get_option(parser, "threads", 2);
    double nu = get_option(parser, "nu", 0.1);
    int oversampling = get_option(parser, "oversampling", 20);
    int cascade = get_option(parser, "cascade", 10);
    int testSplitsParam = get_option(parser, "test-splits", 20);
    int featuresPoolSizeParam = get_option(parser, "features-pool-size", 400);

    if (treeDepthParam < 2)
      treeDepthParam = 2;

    if (threadsParam < 2)
      threadsParam = 2;
    // The faces directory contains a training dataset and a separate
    // testing dataset.  The training data consists of 4 images, each
    // annotated with rectangles that bound each human face along with 68
    // face landmarks on each face.  The idea is to use this training data
    // to learn to identify the position of landmarks on human faces in new
    // images. 
    // 
    // Once you have trained a shape_predictor it is always important to
    // test it on data it wasn't trained on.  Therefore, we will also load
    // a separate testing set of 5 images.  Once we have a shape_predictor 
    // created from the training data we will see how well it works by
    // running it on the testing images. 
    // 
    // So here we create the variables that will hold our dataset.
    // images_train will hold the 4 training images and faces_train holds
    // the locations and poses of each face in the training images.  So for
    // example, the image images_train[0] has the faces given by the
    // full_object_detections in faces_train[0].
    dlib::array<array2d<unsigned char> > images_train, images_test;
    std::vector<std::vector<full_object_detection> > faces_train, faces_test;

    // Now we load the data.  These XML files list the images in each
    // dataset and also contain the positions of the face boxes and
    // landmarks (called parts in the XML file).  Obviously you can use any
    // kind of input format you like so long as you store the data into
    // images_train and faces_train.  But for convenience dlib comes with
    // tools for creating and loading XML image dataset files.  Here you see
    // how to load the data.  To create the XML files you can use the imglab
    // tool which can be found in the tools/imglab folder.  It is a simple
    // graphical tool for labeling objects in images.  To see how to use it
    // read the tools/imglab/README.txt file.
    cout << "Loading training images" << endl;
    load_image_dataset(images_train, faces_train, dataset_directory + "/training_with_face_landmarks.xml");
    cout << "Loading testing images" << endl;
    load_image_dataset(images_test, faces_test, dataset_directory + "/testing_with_face_landmarks.xml");

    // Now make the object responsible for training the model.  
    shape_predictor_trainer trainer;
    // This algorithm has a bunch of parameters you can mess with.  The
    // documentation for the shape_predictor_trainer explains all of them.
    // You should also read Kazemi's paper which explains all the parameters
    // in great detail.  However, here I'm just setting three of them
    // differently than their default values.  I'm doing this because we
    // have a very small dataset.  In particular, setting the oversampling
    // to a high amount (300) effectively boosts the training set size, so
    // that helps this example.
    trainer.set_oversampling_amount(oversampling);
    // I'm also reducing the capacity of the model by explicitly increasing
    // the regularization (making nu smaller) and by using trees with
    // smaller depths.  
    trainer.set_nu(nu);
    trainer.set_tree_depth(treeDepthParam);
    trainer.set_cascade_depth(cascade);
    trainer.set_num_test_splits(testSplitsParam);
    trainer.set_feature_pool_size(featuresPoolSizeParam);

    // some parts of training process can be parallelized.
    // Trainer will use this count of threads when possible
    trainer.set_num_threads(threadsParam);

    // Tell the trainer to print status messages to the console so we can
    // see how long the training will take.
    trainer.be_verbose();

    // Now finally generate the shape model
    shape_predictor sp = trainer.train(images_train, faces_train);


    // Now that we have a model we can test it.  This function measures the
    // average distance between a face landmark output by the
    // shape_predictor and where it should be according to the truth data.
    // Note that there is an optional 4th argument that lets us rescale the
    // distances.  Here we are causing the output to scale each face's
    // distances by the interocular distance, as is customary when
    // evaluating face landmarking systems.
    cout << "mean training error: " <<
      test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << endl;

    // The real test is to see how well it does on data it wasn't trained
    // on.  We trained it on a very small dataset so the accuracy is not
    // extremely high, but it's still doing quite good.  Moreover, if you
    // train it on one of the large face landmarking datasets you will
    // obtain state-of-the-art results, as shown in the Kazemi paper.
    cout << "mean testing error:  " <<
      test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << endl;

    // Finally, we save the model to disk so we can use it later.
    serialize(modelFilename) << sp;

    // Write also configuration
    writeConfigToFile(modelFilename + ".cfg", trainer);
  }
  catch (exception& e)
  {
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
  }
}

// ----------------------------------------------------------------------------------------

double interocular_distance(
  const full_object_detection& det
)
{
  dlib::vector<double, 2> l, r;
  double cnt = 0;
  // Find the center of the left eye by averaging the points around 
  // the eye.
  for (unsigned long i = 36; i <= 41; ++i)
  {
    l += det.part(i);
    ++cnt;
  }
  l /= cnt;

  // Find the center of the right eye by averaging the points around 
  // the eye.
  cnt = 0;
  for (unsigned long i = 42; i <= 47; ++i)
  {
    r += det.part(i);
    ++cnt;
  }
  r /= cnt;

  // Now return the distance between the centers of the eyes
  return length(l - r);
}

std::vector<std::vector<double> > get_interocular_distances(
  const std::vector<std::vector<full_object_detection> >& objects
)
{
  std::vector<std::vector<double> > temp(objects.size());
  for (unsigned long i = 0; i < objects.size(); ++i)
  {
    for (unsigned long j = 0; j < objects[i].size(); ++j)
    {
      temp[i].push_back(interocular_distance(objects[i][j]));
    }
  }
  return temp;
}

// ----------------------------------------------------------------------------------------


