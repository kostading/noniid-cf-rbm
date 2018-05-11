
package com.cinescope.cf.layer;

import java.util.Random;

import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Math.Matrix;

/**
 *
 * @author chungb
 */
public class UserLinearLayer extends Layer {
    
    /*
     * @param numUnits  Constructor. Creates a new instance of LinearLayer with 
     *                  'numUnits' units.
     */
    public UserLinearLayer(int numUnits) {
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
            for (int j = 0; j < this.getNumUnits(); j++) {
                activation_probabilities[ithDataVector][j] += this.biases[j];
            }
        }
        return activation_probabilities;
    }
    
    
    public double[][] generateData(double[][] activation_probabilities) {
    	double[][] data = new double[activation_probabilities.length][activation_probabilities[0].length];
    	
    	for (int i=0; i<activation_probabilities.length; i++) {
        	for (int j=0; j<activation_probabilities[i].length; j++) {
        		data[i][j] = Math.round(activation_probabilities[i][j]);
        	}
    	}
    	
        return data;
    }

    public UserLinearLayer clone() {
        return (UserLinearLayer) super.clone();
    }
}
