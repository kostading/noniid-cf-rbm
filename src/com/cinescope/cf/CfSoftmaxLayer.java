package com.cinescope.cf;

import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.LinearLayer;

public class CfSoftmaxLayer extends Layer {

    /*
     * @param numUnits  Constructor. Creates a new instance of LinearLayer with 
     *                  'numUnits' units.
     */
    public CfSoftmaxLayer(int numUnits) {
        super(numUnits);
    }
    
    public double[] getLayerDerivative(double[] layerActivities) {
        double[] derivative = new double[this.getNumUnits()];
        java.util.Arrays.fill(derivative, 1);
        return derivative;
    }
    
    public double[][] getActivationProbabilities(double[][] productSum) {
        double[][] activation_probabilities = productSum;
        for (int ithDataVector = 0; ithDataVector < activation_probabilities.length; ithDataVector++) {  // for each data vector
            for (int j = 0; j < this.getNumUnits(); j+=5) {
            	double denominator = 0;
                for (int k=j; k<j+5; k++) {
                    // e^(activation)
                    activation_probabilities[ithDataVector][k] = Math.exp(activation_probabilities[ithDataVector][k] + this.biases[k]);
                    denominator += activation_probabilities[ithDataVector][k]; // sum[e^(activation)]
                }
                for (int k=j; k<j+5; k++) {
                    // e^(activation) / sum[e^(activation)]
                    activation_probabilities[ithDataVector][k] /= denominator;
                }
            }
        }
        return activation_probabilities;
    }
    
    
    public double[][] generateData(double[][] activation_probabilities) {
    	double[][] data = new double[activation_probabilities.length][activation_probabilities[0].length];
    	
    	for (int i=0; i<activation_probabilities.length; i++) {
        	for (int j=0; j<activation_probabilities[i].length; j+=5) {
        		int maxIndex = j;
        		double maxValue = 0;
        		for (int k=j; k<j+5; k++) {
        			data[i][k] = 0;
        			if (activation_probabilities[i][k] > maxValue) {
        				maxValue = activation_probabilities[i][k];
        				maxIndex = k;
        			}
        		}
        		if (maxValue > 0) {
        			data[i][maxIndex] = 1;
        		}
        	}
    	}
    	
        return data;
    }

    public CfSoftmaxLayer clone() {
        return (CfSoftmaxLayer) super.clone();
    }
}
