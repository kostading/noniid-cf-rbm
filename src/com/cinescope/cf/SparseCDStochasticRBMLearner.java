/*
 * CDStochasticRBMLearner.java
 *
 * Created on October 18, 2007, 2:26 PM
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
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

import com.syvys.jaRBM.RBM;
import com.syvys.jaRBM.Math.Matrix;

/**
 * A modification to the Contrastive Divergence (CD) RBM learner that works with sparse
 * data i.e. user's vectors having missing ratings. Note that the code is copied from jaRBM
 * original implementation and modified to handle the missing ratings properly in order
 * to make the network able to predict them.
 */
public class SparseCDStochasticRBMLearner {
    
    /**
     * Creates a new instance of CDStochasticRBMLearner
     */
    public SparseCDStochasticRBMLearner() {
    }
    
    public static double Learn(RBM rbm, double[] data) {
        double[][] batchData = {data};
        return Learn(rbm, batchData);
    }
    
    public static double Learn(RBM rbm, double[][] data) {
        return Learn(rbm, data, 1);
    }
    
    public static double Learn(RBM rbm, double[][] data, int numGibbsIterations) {
        double[][] hiddenActivities = rbm.getHiddenActivitiesFromVisibleData(data);
        double[][] hiddenData = rbm.GenerateHiddenUnits(hiddenActivities);
        // negative phase
        double[][] negPhaseVisible = rbm.getVisibleActivitiesFromHiddenData(hiddenData);
        // null values missing in original data
        RbmBasedCF.nullMissingValues(data, negPhaseVisible);
        double[][] negPhaseHidden = rbm.getHiddenActivitiesFromVisibleData(negPhaseVisible);
        
        /*
         * CD Procedure with multiple rounds of Gibbs sampling
         * ===================================================
         * Repeat 'n' times: generate hidden actitives from visible, then visible from hidden
         * double[][] weightUpdates = getConnectionWeightUpdates(rbm, data, nthVisible, 
         *                                                            generatedData, nthHiddenActivities);
         * 
         * updateWeights(rbm, weightUpdates);
         * 
         * - May need to rework this class to enable this feature in other 
         *   CD learners in the com.syvys.jaRBM.RBMLearn package.
         * 
         */
        if (numGibbsIterations > 1) {
            for (int gibbsIter = 1; gibbsIter < numGibbsIterations; gibbsIter++) {
                negPhaseHidden = rbm.getHiddenActivitiesFromVisibleData(negPhaseVisible);
                negPhaseVisible = rbm.getVisibleActivitiesFromHiddenData(negPhaseHidden);
                // null values missing in original data
                RbmBasedCF.nullMissingValues(data, negPhaseVisible);
            }
        }
        
        
        // Some RBMs won't be able to return square matrices for weight updates 
        //      and so on (FactoredRBM is one example; it would return a *gigantic* block-diagonal matrix).
        // So, if they can't, they'll throw an UnsupportedOperationException.
        // Once this exception is thrown, we then proceed to try and update the weights using vectors.
        // Using vectors will result in a pretty big performance hit if there are LOTS of weights, 
        //      but the hope is that the vector would be *much* smaller than the full-matrix alternative.
        try {
            double[][] weightUpdates = getConnectionWeightUpdates(rbm, data, hiddenActivities, negPhaseVisible, negPhaseHidden);
            updateWeights(rbm, weightUpdates);
            rbm.UpdateHiddenBiases(hiddenActivities, negPhaseHidden);
            rbm.UpdateVisibleBiases(data, negPhaseVisible);
            
        } catch (UnsupportedOperationException unSupportedEx) {
            try {
                double[] weightUpdates = getConnectionWeightUpdatesVector(rbm, data, hiddenActivities, negPhaseVisible, negPhaseHidden);
                updateWeights(rbm, weightUpdates);
                rbm.UpdateHiddenBiases(hiddenActivities, negPhaseHidden);
                rbm.UpdateVisibleBiases(data, negPhaseVisible);
            } catch (UnsupportedOperationException ex) {
                throw new UnsupportedOperationException("CDStochasticRBMLearner.Learn(RBM, double[][]) : " + ex);
            }
        }
        
        hiddenActivities = rbm.getHiddenActivitiesFromVisibleData(data);
        negPhaseVisible = rbm.getVisibleActivitiesFromHiddenData(hiddenActivities);
        // null values missing in original data
        RbmBasedCF.nullMissingValues(data, negPhaseVisible);
        return Matrix.getMeanSquaredError(data, negPhaseVisible);
    }
    
    public static double[][] getConnectionWeightUpdates(RBM rbm, double[][] data, double[][] hiddenActivities, 
                                double[][] generatedData, double[][] negativePhaseHiddenActivities) {
        int numVisibleUnits = rbm.getNumVisibleUnits();
        int numHiddenUnits = rbm.getNumHiddenUnits();
        //////////////////////////////////////////////////////////////
        //          Updating Symmetric Weights
        //////////////////////////////////////////////////////////////
        double[][] weightUpdates = new double[numVisibleUnits][numHiddenUnits];
        
        final double[][] positivePhaseProduct = rbm.getVisibleHiddenStateProducts(data, hiddenActivities);
        final double[][] negativePhaseProduct = rbm.getVisibleHiddenStateProducts(generatedData, negativePhaseHiddenActivities);
        
        final double[][] previousWeightIncrement = rbm.getConnectionWeightIncrement();
        final double[][] weightConnections = rbm.getConnectionWeights();
        final double momentum = rbm.getMomentum();
        final double learningRate = rbm.getLearningRate();
        final double weightCost = rbm.getWeightCost();
        for (int v = 0; v < numVisibleUnits; v++) {
            for (int h = 0; h < numHiddenUnits; h++) {
                // Calculating how much to change each symmetric weight by.
                weightUpdates[v][h] = momentum * previousWeightIncrement[v][h] +
                                      learningRate * ( (positivePhaseProduct[v][h] - negativePhaseProduct[v][h]) / data.length -
                                      weightCost * weightConnections[v][h]);
            }
        }
        return weightUpdates;
    }
    
    public static double[] getConnectionWeightUpdatesVector(RBM rbm, double[][] data, double[][] hiddenActivities, 
                                double[][] generatedData, double[][] negativePhaseHiddenActivities) {
        // Updating Weights
        double[] weightUpdates = new double[rbm.getConnectionWeightVectorLength()];
        
        final double[] positivePhaseProduct = rbm.getVisibleHiddenStateProductsVector(data, hiddenActivities);
        final double[] negativePhaseProduct = rbm.getVisibleHiddenStateProductsVector(generatedData, negativePhaseHiddenActivities);
        
        final double[] previousWeightIncrement = rbm.getConnectionWeightIncrementVector();
        final double[] weightConnections = rbm.getConnectionWeightVector();
        final double momentum = rbm.getMomentum();
        final double learningRate = rbm.getLearningRate();
        final double weightCost = rbm.getWeightCost();
        for (int w = 0; w < positivePhaseProduct.length; w++) {
                // Calculating how much to change each symmetric weight by.
                weightUpdates[w] = momentum * previousWeightIncrement[w] +
                                      learningRate * ( (positivePhaseProduct[w] - negativePhaseProduct[w]) / data.length -
                                      weightCost * weightConnections[w]);
        }
        return weightUpdates;
    }
    
    public static void updateWeights (RBM rbm, double[] weightUpdatesVector) {
        //////////////////////////////////////////////////////////////
        //          Updating Symmetric Weights
        //////////////////////////////////////////////////////////////
        double[] newWeights = rbm.getConnectionWeightVector();
        for (int w = 0; w < newWeights.length; w++) {
            newWeights[w] += weightUpdatesVector[w];
        }
        rbm.setConnectionWeightVector(newWeights);
        rbm.setConnectionWeightIncrementVector(weightUpdatesVector);
    }
    public static void updateWeights(RBM rbm, double[][] weightUpdates) {
        double[][] weights = rbm.getConnectionWeights();
        for (int v = 0; v < weights.length; v++) {
            for (int h = 0; h < weights[0].length; h++) {
                weights[v][h] += weightUpdates[v][h];
            }
        }
        rbm.setConnectionWeightIncrement(weightUpdates);
    }
    
}
