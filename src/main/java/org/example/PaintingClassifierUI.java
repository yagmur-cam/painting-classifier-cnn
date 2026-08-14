package org.example;

import javax.swing.*;
import java.awt.*;

public class PaintingClassifierUI extends JFrame {
    private static final String[] STYLES = {
                "Baroque", "Cubism", "Expressionism", "Impressionism",
            "Renaissance", "Romanticism", "Surrealism"
    };
    private static final String[] STYLE_DESCRIPTIONS = {
            "Dramatic contrast between light and shadow, rich deep colors, 17th century European art.",
            "Geometric fragmentation, no deep shadows or gradients indicating 3D depth, early 20th century.",
            "Distorted forms, abnormal colors, emotional intensity, early 20th century.",
            "Visible brushstrokes,  pure color sitting next to each other rather than blended, natural light, late 19th century French.",
            "Smooth blending, dark backgrounds, religious compositions, 14th-17th century.",
            "Dramatic landscapes against tiny human figures, emotional subjects, early 19th century.",
            "Technically realistic but impossible imagery, 20th century movement.",
    };

    public PaintingClassifierUI() {
        setTitle("Painting Style Classifier");
        setSize(700, 580);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        getContentPane().setBackground(Color.WHITE);

        JButton uploadButton = new JButton("Upload Painting");
        uploadButton.setBackground(new Color(30, 158, 117));
        uploadButton.setForeground(Color.WHITE);
        uploadButton.setFocusPainted(false);
        uploadButton.setBorderPainted(false);
        uploadButton.setFont(new Font("SansSerif", Font.PLAIN, 14));
        uploadButton.setCursor(new Cursor(Cursor.HAND_CURSOR));

        JLabel imageLabel = new JLabel("No image selected", JLabel.CENTER);
        imageLabel.setBackground(new Color(245, 244, 240));
        imageLabel.setOpaque(true);

        JLabel resultLabel = new JLabel("Predicted style will appear here", JLabel.CENTER);
        JLabel confidenceLabel = new JLabel("Confidence will appear here", JLabel.CENTER);
        JLabel infoLabel = new JLabel("Style info will appear here", JLabel.CENTER);
        JLabel top3Label = new JLabel("Top 3 will appear here", JLabel.CENTER);

        resultLabel.setFont(new Font("SansSerif", Font.BOLD, 15));
        confidenceLabel.setFont(new Font("SansSerif", Font.PLAIN, 13));
        confidenceLabel.setForeground(new Color(15, 110, 86));
        top3Label.setFont(new Font("SansSerif", Font.PLAIN, 12));
        top3Label.setForeground(new Color(100, 100, 100));
        infoLabel.setFont(new Font("SansSerif", Font.ITALIC, 12));
        infoLabel.setForeground(new Color(100, 100, 100));


        uploadButton.addActionListener( e -> {
            JFileChooser fileChooser = new JFileChooser();
            int result = fileChooser.showOpenDialog(this);
            if (result == JFileChooser.APPROVE_OPTION) {
                String imagePath = fileChooser.getSelectedFile().getAbsolutePath();
                ImageIcon icon = new ImageIcon(new ImageIcon(imagePath)
                        .getImage().getScaledInstance(300, 300, Image.SCALE_SMOOTH));
                imageLabel.setIcon(icon);
                imageLabel.setText("");
                try {
                    Evaluator evaluator = new Evaluator();
                    double[] probs = evaluator.predictProbabilities(imagePath);
                    int topIndex = 0;
                    for (int i = 1; i < probs.length; i++) {
                        if (probs[i] > probs[topIndex]) {
                            topIndex = i;
                        }
                    }
                    int topConfidence = (int) Math.round(probs[topIndex] * 100); //calculate the percentage
                    resultLabel.setText("Predicted Style: " + STYLES[topIndex]);
                    confidenceLabel.setText("Confidence: " + topConfidence + "%");
                    infoLabel.setText("<html><body>" + STYLE_DESCRIPTIONS[topIndex] + "</body></html>");
                    int first = topIndex;
                    int second = 0;
                    int third = 0;

                    for (int i = 0; i < probs.length; i++) {
                        if (i == first) continue;
                        if (probs[i] > probs[second] || second == first) second = i;
                    }
                    for (int i = 0; i < probs.length; i++) {
                        if (i == first || i == second) continue;
                        if (probs[i] > probs[third] || third == first || third == second) third = i;
                    }
                    int conf2 = (int) Math.round(probs[second] * 100);
                    int conf3 = (int) Math.round(probs[third] * 100);
                    top3Label.setText("Top 3: " + STYLES[first] + " " + topConfidence + "% | " +
                            STYLES[second] + " " + conf2 + "% | " +
                            STYLES[third] + " " + conf3 + "%");
                } catch (Exception ex) {
                    resultLabel.setText("Error: " + ex.getMessage());
                }
            }
        });

        add(uploadButton, BorderLayout.NORTH);
        add(imageLabel, BorderLayout.CENTER);
        JPanel bottomPanel = new JPanel();
        bottomPanel.setBackground(Color.WHITE);
        bottomPanel.setBorder(BorderFactory.createEmptyBorder(12, 16, 12, 16));
        bottomPanel.setLayout(new BoxLayout(bottomPanel, BoxLayout.Y_AXIS));
        bottomPanel.add(resultLabel);
        bottomPanel.add(confidenceLabel);
        bottomPanel.add(top3Label);
        bottomPanel.add(infoLabel);
        add(bottomPanel, BorderLayout.SOUTH);
        setVisible(true);
    }

    public static void main(String[] args) {
        new PaintingClassifierUI();
    }
}

