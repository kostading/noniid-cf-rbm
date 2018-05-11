package com.cinescope.cf;

import com.syvys.jaRBM.*;
import com.syvys.jaRBM.RBMLearn.BackpropRBMLearner;
import com.syvys.jaRBM.RBMNetLearn.RBMNetLearner;
import com.syvys.jaRBM.Math.Matrix;
import com.syvys.jaRBM.IO.BatchDatasourceReader;
import java.util.Arrays;

/**
 * A modification to the backpropagation RBM learner that works with sparse data
 * i.e. user's vectors having missing ratings. Note that the code is copied from jaRBM
 * original implementation and modified to handle the missing ratings properly in order
 * to make the network able to predict them.
 */
public class SparseBackpropLearner extends RBMNetLearner {

    protected BatchDatasourceReader _targetReader;
    
    protected double[] _weights;
    protected double[] _gradient;
    protected double[] _weightIncrement;
    
    protected double _learningRate;
    protected double _weightCost;
    protected double _momentum;
    /**
     * Creates a new instance of BackpropLearner
     */
    public SparseBackpropLearner(RBMNet myrbmnet, BatchDatasourceReader dataReader, 
                           BatchDatasourceReader targetReader, int batchSize) {
        super(myrbmnet, dataReader, batchSize);
        _targetReader = targetReader;
        
        _weights = myrbmnet.getConnectionWeightVector();
        _gradient = new double[_weights.length];
        _weightIncrement = new double[_weights.length];
        
        _learningRate = _learnedRBMNet.getRBM(0).getLearningRate();
        _weightCost = _learnedRBMNet.getRBM(0).getWeightCost();
        _momentum = _learnedRBMNet.getRBM(0).getMomentum();
        
        this._targetReader.beforeFirst();
        this._batchReader.beforeFirst();
        
        int vectorLength = this._batchReader.getNext().length;
        
        if (_learnedRBMNet.getNumOfBottomVisibleUnits() != vectorLength) {
            System.out.println("Error -> BackpropLearner.Learn(): The number bottom-level visible units is not equal to the length of the data vectors.");
            System.out.println("rbmnet has " + _learnedRBMNet.getNumOfBottomVisibleUnits() + 
                    " and data is " + vectorLength);
        }
        // Let's make sure we're at the beginning.
        this._batchReader.beforeFirst();
    }
    
    public double Learn() {
        this._targetReader.beforeFirst();
        this._batchReader.beforeFirst();

        double error = 0.0;
        
        int numBatches = 0;
        for (numBatches = 0; this._batchReader.hasNext() && this._targetReader.hasNext(); numBatches++) {
            double[][] inputBatch = this._batchReader.getNext(this._batchSize);
            double[][] targetBatch = this._targetReader.getNext(this._batchSize);

            if (inputBatch.length != targetBatch.length) {
                int minBatchSize = Math.min(inputBatch.length, targetBatch.length);
                if (minBatchSize == 0) { 
                    continue;
                }
                if (inputBatch.length == minBatchSize) {
                    targetBatch = Arrays.copyOf(targetBatch, minBatchSize);
                } else {
                    inputBatch = Arrays.copyOf(inputBatch, minBatchSize);
                }
            }
            
            double batcherror = this.Learn(inputBatch, targetBatch);
            error += batcherror;

            if (Double.isNaN(error)) {
                System.out.println(this.getClass().getSimpleName() + ".Learn(): error just turned into NaN on batch "+ numBatches+"!");
                break;
            }
        }
        // return the average error per batch of learning the training dataset.
        return error / numBatches;
    }
    
    public double Learn(double[] data, double[] target) {
        double[][] batchData = {data};
        double[][] batchTarget = {target};
        return Learn(batchData, batchTarget);
    }
    
    public double Learn(double[][] batchData, double[][] batchTargets) {
        _weightIncrement = calcGradients(batchData, batchTargets);
        
        for (int i = 0; i < _weights.length; i++) {
            _weightIncrement[i] = -_weightIncrement[i];
            _weightIncrement[i] = _momentum * _weightIncrement[i] + 
                                  _learningRate * (_weightIncrement[i] - _weightCost * _weights[i]);
            _weights[i] += _weightIncrement[i];
        }

        _learnedRBMNet.setConnectionWeightVector(_weights.clone());

        double[][] output = _learnedRBMNet.getHiddenActivitiesFromVisibleData(batchData);
        RbmBasedCF.nullMissingValues(batchData, output);
        
        return Matrix.getMeanSquaredError(output, batchTargets);
    }
    
    protected double[] calcGradients(double[][] batchData, double[][] batchTargets) {
        double[] dEdw = new double[_learnedRBMNet.getConnectionWeightVectorLength()];
        
        double[][][] hiddenActivities = new double[_learnedRBMNet.getNumRBMs()][batchData.length][];
        
        // propagating the data forward through the neural net.
        double[][] workingData = batchData;
        for (int ithRBM = 0; ithRBM < _learnedRBMNet.getNumRBMs(); ithRBM++) {
            // positive phase
            hiddenActivities[ithRBM] = _learnedRBMNet.getRBM(ithRBM).getHiddenActivitiesFromVisibleData(workingData);
            if (ithRBM == (_learnedRBMNet.getNumRBMs() - 1)) {
                RbmBasedCF.nullMissingValues(batchData, hiddenActivities[ithRBM]);
            }
            workingData = hiddenActivities[ithRBM];
        }
        
        // Calculating the error for each data vector in the batch
        double[][] output = workingData; // <- The batch output! output[i].length should == batchTargets[i].length
        double[][] error = new double[batchData.length][output[0].length];
        for (int ibatch = 0; ibatch < batchData.length; ibatch++) {
            for (int i = 0; i < output[0].length; i++) {
            	if (batchTargets[ibatch][i] != 0.0) {
            		error[ibatch][i] = (batchTargets[ibatch][i] - output[ibatch][i]); // <- error.;
            	}
            }
        }
        
        int weightvectorindex = dEdw.length - 1;
        // propagating the errors backward through the neural net.
        for (int ithRBM = _learnedRBMNet.getNumRBMs() - 1; ithRBM >= 0 ; ithRBM--) {
            double[] ithRBMweightDerivatives;
            if (ithRBM == 0) {
                ithRBMweightDerivatives = _learnedRBMNet.getRBM(ithRBM).getConnectionWeightDerivativesVector( 
                                                                                  batchData, 
                                                                                  hiddenActivities[ithRBM], error);
            } else {
                ithRBMweightDerivatives = _learnedRBMNet.getRBM(ithRBM).getConnectionWeightDerivativesVector(
                                                                                  hiddenActivities[ithRBM - 1],
                                                                                  hiddenActivities[ithRBM], error);
            }
            // 'error' is error vector for the next layer down.
            error = BackpropRBMLearner.getVisibleLayerError(_learnedRBMNet.getRBM(ithRBM), error);
            
            // Tack on the weight derivatives onto dEdw!!
            double[] rbmgradientvector = ithRBMweightDerivatives;
            for (int i = rbmgradientvector.length - 1; i >= 0; i--) {
                dEdw[weightvectorindex] = rbmgradientvector[i];
                weightvectorindex--;
            }
        }
        
        // Normalize the gradients.
        double factor = -2D / batchData.length / _learnedRBMNet.getNumOfTopHiddenUnits();
        for (int i = 0; i < dEdw.length; i++ ) {
            dEdw[i] *= factor;
        }
        return dEdw;
    }
    
}
