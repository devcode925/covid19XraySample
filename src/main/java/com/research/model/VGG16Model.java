package com.research.model;

public class VGG16Model {
/*
   public static void main(String[] args) throws IOException {
        ZooModel zooModel = new VGG16();
        LOGGER.info("Start Downloading VGG16 model...");
        ComputationGraph preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        LOGGER.info(preTrainedNet.summary());

        LOGGER.info("Start Downloading Data...");

        downloadAndUnzipDataForTheFirstTime();
        LOGGER.info("Data unzipped");
        // Define the File Paths
        File trainData = new File(TRAIN_FOLDER);
        File testData = new File(TEST_FOLDER);
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);


        InputSplit[] sample = train.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
        DataSetIterator trainIterator = getDataSetIterator(sample[0]);
        DataSetIterator devIterator = getDataSetIterator(sample[1]);


        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(5e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(seed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(FREEZE_UNTIL_LAYER)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(NUM_POSSIBLE_LABELS)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), FREEZE_UNTIL_LAYER)
                .build();
        vgg16Transfer.setListeners(new ScoreIterationListener(5));
        LOGGER.info(vgg16Transfer.summary());

        DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0]);
        int iEpoch = 0;
        int i = 0;
        while (iEpoch < EPOCH) {
            while (trainIterator.hasNext()) {
                DataSet trained = trainIterator.next();
                vgg16Transfer.fit(trained);
                if (i % SAVING_INTERVAL == 0 && i != 0) {

                    ModelSerializer.writeModel(vgg16Transfer, new File(SAVING_PATH + i + "_epoch_" + iEpoch + ".zip"), false);
                    evalOn(vgg16Transfer, devIterator, i);
                }
                i++;
            }
            trainIterator.reset();
            iEpoch++;

            evalOn(vgg16Transfer, testIterator, iEpoch);
        }
    }

 */
}
