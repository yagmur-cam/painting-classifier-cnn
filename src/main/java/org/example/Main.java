package org.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class Main {
    public static void main(String[] args) throws Exception {
        ModelBuilder builder = new ModelBuilder();
        MultiLayerConfiguration conf = builder.buildModel();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println("Model initialized");

        String datasetPath = "dataset";
        Trainer trainer = new Trainer();
        trainer.trainModel(model, datasetPath);
    }
}
