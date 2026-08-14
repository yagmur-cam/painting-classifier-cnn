package org.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class Main {
    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "train";
        String datasetpath = "dataset";

        switch (mode) {
            case "train":
                ModelBuilder builder = new ModelBuilder();
                MultiLayerConfiguration conf = builder.buildModel();
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                System.out.println("Model initialized");
                new Trainer().trainModel(model, datasetpath);
                break;
            case "evaluate":
                new Evaluator().evaluate(datasetpath);
                break;
            case "predict":
                if (args.length < 2) {
                    System.out.println("Usage: predict <path-to-image>");
                    return;
                }
                System.out.println("Predicted style: " + new Evaluator().predict(args[1]));
                break;
            case "ui":
                new PaintingClassifierUI();
                break;
            default:
                System.out.println("Usage: train | evaluate | predict <image>");
        }
    }
}
