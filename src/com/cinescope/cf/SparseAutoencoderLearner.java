package com.cinescope.cf;

import com.syvys.jaRBM.*;
import com.syvys.jaRBM.Math.Matrix;
import com.syvys.jaRBM.IO.BatchDatasourceReader;

/**
 * A modification to the auto-encoder RBM learner that works with sparse data
 * i.e. user's vectors having missing ratings. Note that the code is copied from jaRBM
 * original implementation and modified to handle the missing ratings properly in order
 * to make the network able to predict them.
 */
public class SparseAutoencoderLearner extends SparseBackpropLearner {
    
    /**
     * Creates a new instance of AutoencoderLearner
     */
    /*
     *
     * The number of top-most hidden units doesn't have to match the number of 
     * bottom-most visible units; this flips it automatically.
     *
     */
    public SparseAutoencoderLearner(RBMNet myrbmnet, BatchDatasourceReader dataReader, int batchSize) {
        // stick an upside-down version of itself on top.
        super(myrbmnet.FlipAndStack(), dataReader, dataReader.clone(), batchSize);
        
        this._batchReader.beforeFirst();
        if (_learnedRBMNet.getNumOfBottomVisibleUnits() != this._batchReader.getNext().length) {
            System.out.println("Error: AutoencoderLearner: The number bottom-level visible units is not equal to the length of the data vectors.");
        }
        this._batchReader.beforeFirst();
    }
    
    // Cut the RBM in half!... to return the original RBMNet that the user passed in.
    public RBMNet getLearnedRBMNet() {
        RBMNet myrbmnet = new RBMNet(_learnedRBMNet.getRBM(0).clone());
        int lastRBM = _learnedRBMNet.getNumRBMs()/2;
        for (int i = 1; i < lastRBM; i++) {
            try {
                myrbmnet.AddRBM(_learnedRBMNet.getRBM(i).clone());
            } catch (Exception e) {
                System.out.println("AutoencoderLearner.getLearnedRBMNet(): " + e.getMessage());
            }
        }
        return myrbmnet;
    }
    
    public double getError(double[][] data) {
        return Matrix.getMeanSquaredError(super._learnedRBMNet.getHiddenActivitiesFromVisibleData(data), data);
    }
    
}
    
