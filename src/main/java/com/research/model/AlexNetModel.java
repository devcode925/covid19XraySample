package com.research.model;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class AlexNetModel {
    private static final Logger logger = LoggerFactory.getLogger(AlexNetModel.class);

    public MultiLayerNetwork getAlexNetModel( ModelData modelData ) {

       if( modelData == null )
           return null;
        /*
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         */
        double nonZeroBias = 1;
        double dropOut = 0.5;
        logger.info("** Get Model before builder.");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(modelData.getSeed())
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new AdaDelta())
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                .list()
                .layer(convInit("cnn1", modelData.getChannels(), 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(maxPool("maxpool1", new int[]{3, 3}))
                .layer(conv5x5(256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
                .layer(new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(maxPool("maxpool2", new int[]{3, 3}))
                .layer(conv3x3("cnn3", 384, 0))
                .layer(conv3x3("cnn4", 384, nonZeroBias))
                .layer(conv3x3("cnn5", 256, nonZeroBias))
                .layer(maxPool("maxpool3", new int[]{3, 3}))
                .layer(fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new NormalDistribution(0, 0.005)))
                .layer(fullyConnected("ffn2", 500, nonZeroBias, dropOut, new NormalDistribution(0, 0.005)))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(modelData.getNumberOfLabels())
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(modelData.getHeight(), modelData.getWidth(), modelData.getChannels()))
                .build();

        return new MultiLayerNetwork(conf);

    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name("cnn2").nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).weightInit(new WeightInitDistribution(dist)).build();
    }

}
