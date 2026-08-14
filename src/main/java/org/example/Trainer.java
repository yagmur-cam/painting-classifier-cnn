package org.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

public class Trainer {
    public void trainModel (MultiLayerNetwork model, String datasetPath) throws IOException {
        int NUM_EPOCHS = 30;
        DataLoader data = new DataLoader(datasetPath);
        DataSetIterator trainIterator = data.loadData("train");
        DataSetIterator validationIterator = data.loadData("validation");
        DataSetIterator testIterator = data.loadData("test");

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            model.fit(trainIterator);
            Evaluation eval = model.evaluate(validationIterator);
            System.out.println("Epoch " + epoch + " - Accuracy: " + eval.accuracy());
            validationIterator.reset();
            trainIterator.reset();
        }
        ModelSerializer.writeModel(model, new File("painting-classifier-model.zip"), true);
        System.out.println("Training complete. Model saved.");
        Evaluation testEval = model.evaluate(testIterator);
        System.out.println("Test Accuracy: " + testEval.accuracy());
        System.out.println(testEval.stats());
    }
}
