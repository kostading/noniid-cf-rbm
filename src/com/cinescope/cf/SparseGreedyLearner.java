/*
 * GreedyLearner.java
 *
 * Created on January 23, 2007, 8:26 PM
 *
 */
/************* License relating to this Java implementation **********
 * This is a Java implementation of a Restricted Boltzmann Machine.
 * The GPL license applies to this Java implementation only.
 *
 * Copyright (C) 2006, 2007 Benjamin Chung   [benchung AT NO $PAM gmail DOT com]
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 ********************************************************************* */


package com.cinescope.cf;

import com.syvys.jaRBM.*;
import com.syvys.jaRBM.IO.BatchDatasourceFileWriter;
import com.syvys.jaRBM.RBMLearn.CDStochasticRBMLearner;
import com.syvys.jaRBM.RBMNetLearn.RBMNetLearner;
import com.syvys.jaRBM.IO.BatchDatasourceReader;
import java.io.IOException;
import java.io.File;

/**
 * A modification to the greedy layer-by-layer learning RBM learner that works with sparse
 * data i.e. user's vectors having missing ratings. Note that the code is copied from jaRBM
 * original implementation and modified to handle the missing ratings properly in order
 * to make the network able to predict them.
 */
public class SparseGreedyLearner extends RBMNetLearner {
    
    public static int DEFAULT_NUM_EPOCHS_PER_LAYER = 100;
    
    /**
     * Creates a new instance of GreedyContrastiveDivergenceLearner
     * @param numEpochs The number of times to train the RBMNet on the entire
     *                  dataset (data[][])
     */
    public SparseGreedyLearner(RBMNet myrbmnet, BatchDatasourceReader batchReader, int batchSize) {
        super(myrbmnet, batchReader, batchSize);
    }

    /*
     * Trains this RBMNet with the the provided training dataset using the 
     * greedy layer-by-layer learning algorithm outlined in:
     *  G. E. Hinton, S. Osindero, Y. W. Teh, Neural Computation. 
     *  18, 1527 (2006), and used in:
     *  G. E. Hinton, R. R. Salakhutdinov, Science. 504, Vol313 (2006)
     *
     * @param data  An array of data vectors. The complete training dataset
     *              should be contained within data, with each "i'th" data 
     *              vector stored in data[i].
     * @param batchSize The number of data-vectors each batch used for training 
     *                  should contain. If there are any left over in the last
     *                  batch of data vectors, they are simply bundled together
     *                  and run through the RBMNet as a batch.
     * @returns     The squared error of learning the data this epoch.
     */
    public double Learn() {
        return Learn(DEFAULT_NUM_EPOCHS_PER_LAYER);
    }
    
    public double Learn (int numOfEpochsPerLayer) {
        // Default is to greedy-learn with all RBMs.
        return Learn(numOfEpochsPerLayer, _learnedRBMNet.getNumRBMs() - 1);
    }
    
    public double Learn (int numOfEpochsPerLayer, int upToAndIncludingLayer) {
        int numGibbsIterationsToIncreaseBy = 0; // Perform greedy learning using vanilla CD using one iteration of gibbs sampling
        int numEpochsPerGibbsIncrease = 5000; // for each layer, increase number of gibbs sampling interations every 5000 epochs
        int maxNumTimesToIncreaseGibbsIterations = 0; // Increase gibbs iterations a maximum of 1 time
        return Learn(numOfEpochsPerLayer, upToAndIncludingLayer, numGibbsIterationsToIncreaseBy,
                                          numEpochsPerGibbsIncrease, maxNumTimesToIncreaseGibbsIterations);
    }
    
    public double Learn (int numOfEpochsPerLayer, int upToAndIncludingLayer, 
                         int numGibbsIterationsToIncreaseBy, int numEpochsPerGibbsIncrease, 
                         int maxNumTimesToIncreaseGibbsIterations) {
        
        if (upToAndIncludingLayer < 0 || upToAndIncludingLayer >= _learnedRBMNet.getNumRBMs()) {
            // If upToAndIncludingLayer is not within [0.._learnedRBMNet.getNumRBMs()]... return false.
            return 0.;
        }
        
        BatchDatasourceReader tempReader = this._batchReader; // first, read the data inputs
        int hashCode = this._learnedRBMNet.hashCode() + (int)(Math.random()*1000);
                        //   ^----  pseudo-uniquely identify this rbmnet from all the other ones.
                        //          We'll need this hash to separate this net's temporary activities 
                        //          from those of other RBMNets.
        
        for (int ithRBM = 0; ithRBM <= upToAndIncludingLayer; ithRBM++) {
            tempReader.beforeFirst();
            this.LearnLayer(ithRBM, tempReader, numOfEpochsPerLayer, numGibbsIterationsToIncreaseBy,
                                                numEpochsPerGibbsIncrease, maxNumTimesToIncreaseGibbsIterations);
            try {
                BatchDatasourceFileWriter tempWriter = new BatchDatasourceFileWriter("GreedyLearner_"+hashCode+"_tempActivities_" + ithRBM, false);
                tempReader.beforeFirst();
                // Write the hidden activities to file for a temporary location, so the JVM won't
                // crash if we're working with really large datasets.
                while (tempReader.hasNext()) {
                    tempWriter.append(_learnedRBMNet.getRBM(ithRBM).getHiddenActivitiesFromVisibleData(tempReader.getNext(1000)));
                }
                this._batchReader.beforeFirst();
                if (ithRBM >= 1) {
                    tempReader.close(); // Only close the readers for the temporary inter-layer activities!
                }
                tempReader = tempWriter.getReader(); // set the temporaryBatchReader to read the freshly-written activities.

            } catch (Exception ex) {
                System.err.println("Error -> GreedyLearner.Learn() : " + ex.getMessage());
                ex.printStackTrace();
            }
        }
        
        // Remove the temporary activities file if they exist.
        for (int ithRBM = 0; ithRBM <= upToAndIncludingLayer; ithRBM++) {
            String tempFilename = "GreedyLearner_"+hashCode+"_tempActivities_" + ithRBM;
            File tempFile = new File(tempFilename);
            if (tempFile.exists() && tempFile.canWrite()) {
                tempFile.delete();
            }
        }
        return 1.;
    }
    
    protected double LearnLayer(int ithRBM, BatchDatasourceReader batchReader, int numOfEpochs) {
        return this.LearnLayer(ithRBM, batchReader, numOfEpochs, 0, 5000, 0);
    }
    
    
    protected double LearnLayer(int ithRBM, BatchDatasourceReader batchReader, int numOfEpochs, 
                                int numGibbsIterationsToIncreaseBy, int numEpochsPerGibbsIncrease, 
                                int maxNumTimesToIncreaseGibbsIterations) {
        
        RBM rbm = _learnedRBMNet.getRBM(ithRBM);
        
        double error = 0.;
        double originalMomentum = rbm.getMomentum();
        
        int currentNumGibbsIterations = 1;
        int timesGibbsIncreased = 0;
        
        for (int e = 1; e <= numOfEpochs; e++) {
            if (numOfEpochs <= 5) // halve the momentum initially so initial changes aren't too crazy.
                rbm.setMomentum(originalMomentum / 2.);
            else 
                rbm.setMomentum(originalMomentum);
            
            // Let the first 'numEpochsPerGibbsIncrease' epochs of CD be with
            // one iteration of gibbs sampling
            if (e % numEpochsPerGibbsIncrease == 0) {
                // If we haven't reached our maximum number of times...
                if (timesGibbsIncreased < maxNumTimesToIncreaseGibbsIterations) {
                    timesGibbsIncreased++;
                    // Increase the number of gibbs sampling by 'numGibbsIterationsToIncreaseBy'
                    currentNumGibbsIterations += numGibbsIterationsToIncreaseBy;
                    System.out.println("GreedyLearner.LearnLayer(): layer "+ithRBM+", " +
                                       " epoch " + e + ". Gibbs increase #"+timesGibbsIncreased+",  " +
                                       "increased gibbs iterations by " + numGibbsIterationsToIncreaseBy + 
                                       " to " + currentNumGibbsIterations);
                }
            }
            
            double avgBatchError = 0;
            int numBatches = 0;
            batchReader.beforeFirst(); // reset data cursor to beginning for the next epoch.
            while ( batchReader.hasNext() ) {
                double[][] nextVector = batchReader.getNext(this._batchSize);
//                System.out.println("GreedyLearner.LearnLayer(): nextVector is " + nextVector[0].length + " long");
//                System.out.println("GreedyLearner.LearnLayer(): rbm visible Layer is " + rbm.getNumOfBottomVisibleUnits() + " long");
                
                double batcherror = 0.;
                
                //batcherror += CDStochasticRBMLearner.Learn(rbm, nextVector);
                if (ithRBM == 0) {
                	batcherror += SparseCDStochasticRBMLearner.Learn(rbm, nextVector, currentNumGibbsIterations);
                } else {
                	batcherror += CDStochasticRBMLearner.Learn(rbm, nextVector, currentNumGibbsIterations);
                }
                
                avgBatchError += batcherror;
//                System.out.println("GreedyLearner.LearnLayer(): batch " + numBatches + " batchError = " + batcherror);
                if (Double.isNaN(avgBatchError)) {
                    System.out.println("GreedyLearner.LearnLayer(): nan on batch " + numBatches);
                }
                numBatches++;
            }
            // just in case we want to see the average error per batch for debugging purposes.
            avgBatchError /= (double)numBatches;
            System.out.println("GreedyLearner.LearnLayer(): layer "+ ithRBM + ", epoch " + e + ", error = " + avgBatchError);
            error += avgBatchError;
        }
        return error / numOfEpochs;
    }
    
    // This function trains a specific layer only.
    // Propagates the data up the net until it reaches the RBM you want to trian, 
    // and then trains that RBM on the hidden activities of the previous layer below.
    public double LearnSpecificLayer(int ithLayer, int numOfEpochsPerLayer) {
        BatchDatasourceReader tempReader = this._batchReader; // first, read the data inputs
        int hashCode = this._learnedRBMNet.hashCode() + (int)(Math.random()*1000);
                        //   ^----  pseudo-uniquely identify this rbmnet from all the other ones.
                        //          We'll need this hash to separate this net's temporary activities 
                        //          from those of other RBMNets.
        if (ithLayer > 0) {
            for (int ithRBM = 0; ithRBM < ithLayer; ithRBM++) {
                try {
                    BatchDatasourceFileWriter tempWriter = new BatchDatasourceFileWriter("GreedyLearner_"+hashCode+"_tempActivities_" + ithRBM, false);
                    this._batchReader.beforeFirst();
                    // Write the hidden activities to file for a temporary location, so the JVM won't
                    // crash if we're working with really large datasets.
                    while (this._batchReader.hasNext()) {
                        tempWriter.append(_learnedRBMNet.getRBM(ithRBM).getHiddenActivitiesFromVisibleData(this._batchReader.getNext(1000)));
                    }
                    tempReader = tempWriter.getReader(); // set the temporaryBatchReader to read the freshly-written activities.

                } catch (ArrayIndexOutOfBoundsException ex) {
                    System.err.println("Error -> GreedyLearner.Learn() : " + ex.getMessage());
                    ex.printStackTrace();
                } catch (IOException ex) {
                    System.err.println("Error -> GreedyLearner.Learn() : " + ex.getMessage());
                    ex.printStackTrace();
                }
            }
        }
        
        LearnLayer(ithLayer, tempReader, numOfEpochsPerLayer);
        
        // Remove the temporary activities file if they exist.
        for (int ithRBM = 0; ithRBM < ithLayer; ithRBM++) {
            String tempFilename = "GreedyLearner_"+hashCode+"_tempActivities_" + ithRBM;
            File tempFile = new File(tempFilename);
            if (tempFile.exists() && tempFile.canWrite()) {
                tempFile.delete();
            }
        }
        
        return 1.;
    }
    
    
}
