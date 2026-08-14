package org.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ScheduleType;


public class ModelBuilder {
    public MultiLayerConfiguration buildModel() {
        {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, 0.001, 0.95)))
                    .weightInit(WeightInit.RELU)
                    .list()

                    .layer(0, new ConvolutionLayer.Builder(3, 3)
                            .nIn(3)
                            .stride(1, 1)
                            .padding(1, 1)
                            .nOut(32)
                            .activation(Activation.RELU)
                            .l2(1e-4)
                            .build())

                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())

                    .layer(2, new ConvolutionLayer.Builder(3, 3)
                            .stride(1, 1)
                            .padding(1, 1)
                            .nOut(64)
                            .activation(Activation.RELU)
                            .build())

                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())

                    .layer(4, new ConvolutionLayer.Builder(3, 3)
                            .stride(1, 1)
                            .padding(1, 1)
                            .nOut(128)
                            .activation(Activation.RELU)
                            .build())

                    .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())

                    .layer(6, new ConvolutionLayer.Builder(3, 3)
                            .stride(1, 1)
                            .padding(1, 1)
                            .nOut(256)
                            .activation(Activation.RELU)
                            .build())

                    .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())

                    .layer(8, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())

                    .layer(9, new DenseLayer.Builder()
                            .nOut(256)
                            .activation(Activation.RELU)
                            .l2(1e-4)
                            .build())

                    .layer(10, new DropoutLayer.Builder(0.5).build())
                    .layer(11, new OutputLayer.Builder()
                            .nOut(7)
                            .activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT)
                            .build())
                    .setInputType(InputType.convolutional(128, 128, 3))
                    .build();
            return conf;
        }
    }
}
