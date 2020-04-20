package com.research.only;

import com.research.model.AlexNetModel;
import com.research.model.ModelData;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import static java.lang.Math.toIntExact;
/**
 * for testing Deeplearning4j Convolutional Model using Covid Image Xrays.
 * images from https://github.com/UCSD-AI4H/COVID-CT
 * CITE:
 @Article{he2020sample,
 author  = {He, Xuehai and Yang, Xingyi and Zhang, Shanghang, and Zhao, Jinyu and Zhang, Yichen and Xing, Eric, and Xie,       Pengtao},
 title   = {Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans},
 journal = {medrxiv},
 year    = {2020},
 }
 * @author Chuck Hernandez
 */
public class ResearchXrayCovid19 {
    private static final Logger logger = LoggerFactory.getLogger(ResearchXrayCovid19.class);
    //resize images to 224, set the channel to grayscale and small batches.
    private static final int HEIGHT =244;
    private static final int WIDTH = 244;
    private static final int CHANNELS =1;
    private static final int BATCH_SIZE= 15;
    private static final long SEED_24 = 24;
    private static final Random RANDOM = new Random(SEED_24);
    private static final int EPOCHS = 200;
    private static final int LABEL_COUNT = 2;
    List<String> classLabels;

    public void run(String localPath) {

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(localPath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, RANDOM);

        int numImages = toIntExact(fileSplit.length());
        BalancedPathFilter pathFilter = new BalancedPathFilter(RANDOM, labelMaker, numImages, LABEL_COUNT,18);
        StringBuilder sb = new StringBuilder("***number of images is: ");
        sb.append(numImages).append("** image files loaded from: ").append(mainPath);
        logger.info(sb.toString());

        double splitTrainTest = 0.8;
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1- splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        logger.info("***Getting the model");
        MultiLayerNetwork networkModel = getAlexNetModel();
        networkModel.init();
        logger.info("***Model loaded and initialized.");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        logger.info("***UI Server started and stats attached, about to iterate test data");

        DataSetIterator testIterator =trainData( testData, labelMaker, false);
        //set up the listeners for scoring and UI stats
        networkModel.setListeners(new StatsListener( statsStorage), new ScoreIterationListener(1),
                new EvaluativeListener(Objects.requireNonNull(testIterator), 1, InvocationType.EPOCH_END));

        // Train without transformations
        logger.info("***Calling fit on the model with the training data, without transforms.");
        DataSetIterator trainIterator = trainData(trainData,labelMaker,false);
        networkModel.fit(Objects.requireNonNull(trainIterator), EPOCHS);
       logger.info("***Fit called on training data.");

        // Train with transformations
        logger.info("***Calling fit on the model with the training data, with transforms.");
        trainIterator = trainData(trainData,labelMaker,true);
        networkModel.fit(Objects.requireNonNull(trainIterator), EPOCHS);
        logger.info("***Fit called on training data.");

        //get predict results with trained model.
        trainIterator.reset();
        DataSet testDataSet = trainIterator.next();

        int labelIndex = testDataSet.getLabels().argMax(1).getInt(1);
        int[] predictedClasses = networkModel.predict(testDataSet.getFeatures());
        String expectedResult = getLabels().get(labelIndex);
        String modelPrediction = getLabels().get(predictedClasses[1]);
        logger.info("\nExample that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");
        logger.info("****************Example finished********************");
    }
    /*
     * Data Setup -> normalization and transformation
     */
    private ImageTransform getImageTransform(boolean shuffle) {
        ImageTransform flipTransform1 = new FlipImageTransform(RANDOM);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(RANDOM, 42);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1, 0.9),
                new Pair<>(flipTransform2, 0.8),
                new Pair<>(warpTransform, 0.5));

        logger.info("***getImageTransform() completed.");
        return new PipelineImageTransform(pipeline, shuffle);
    }
    /**
     * Data Setup -> define how to load data into net:
     *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
     *  - dataIter = a generator that only loads one batch at a time into memory to save memory
     *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all EPOCHS
     * see Deeplearning4J documentation.
     */
    private synchronized DataSetIterator trainData( InputSplit data, ParentPathLabelGenerator labelMaker, boolean transform) {
        try {
            // test iterator
            ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
            if( transform ) {
                imageRecordReader.initialize(data, getImageTransform(false));
            } else {
                imageRecordReader.initialize(data);
            }
            setClassLabels(imageRecordReader.getLabels() );

            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABEL_COUNT);
            DataNormalization imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
            imagePreProcessingScaler.fit(dataSetIterator);
            dataSetIterator.setPreProcessor(imagePreProcessingScaler);

            return dataSetIterator;

        } catch (IOException e) {
            logger.error("*** Error in trainData().");
            e.printStackTrace();
            return null;
        }
    }
    private MultiLayerNetwork getAlexNetModel() {
        ModelData modelData = new ModelData(CHANNELS, HEIGHT, WIDTH, LABEL_COUNT, SEED_24);
        AlexNetModel alexNetModel = new AlexNetModel();
        return alexNetModel.getAlexNetModel(modelData);
    }
    private  List<String> getLabels() {
        return classLabels;
    }
    private void setClassLabels( List<String> labels ) {
        classLabels = labels;
    }
    public static void main(String[] args) {
        ResearchXrayCovid19 xrayCovid19 = new ResearchXrayCovid19();
        String imgPath = "C:\\Users\\chernandez\\Documents\\personalEc\\ML\\COVID-CT-master\\Images-processed\\";
        try {
            xrayCovid19.run(imgPath);
            /**
             ZooModel zooModel = VGG16.builder().numClasses(2) .seed(seed) .build();
             zooModel.setInputShape(new int[][]{{ CHANNELS,HEIGHT,WIDTH}});ComputationGraph vgg16 = zooModel.init();
             **/
        }catch( Exception e) {
            System.out.println("Error running: "+ e.getMessage());
        }
    }
}
