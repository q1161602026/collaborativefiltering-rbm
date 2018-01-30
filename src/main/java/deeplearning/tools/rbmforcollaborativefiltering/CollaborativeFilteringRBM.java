/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package deeplearning.tools.rbmforcollaborativefiltering;

import genutils.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import org.jblas.DoubleMatrix;

/**
 *
 * @author thanos
 */
public class CollaborativeFilteringRBM {
    
   
    private static final java.util.logging.Logger _logger = Logger.getLogger(CollaborativeFilteringRBM.class.getName());

    // constants related to the training procedure
    private double lr_w;   // learning rate for weights
    private double lr_visible_b;  // learning rate for biases for visible units
    private double lr_hidden_b;  // learning rate for biases for hidden units
    //double reg_w = 0.0002;
    private double reg_w;
    private double momentumInitial;
    private double momentumFinal;
    private double modifier;

    private DataStructure ds = new DataStructure();

    public CollaborativeFilteringRBM() {
        this(0.1, 0.2, 0.1, 0.002, 0.5, 0.9, 20.0);
    }

    public CollaborativeFilteringRBM(double lr_w, double lr_visible_b, double lr_hidden_b,
                                     double reg_w, double momentumInitial,
                                     double momentumFinal, double modifier) {
        this.lr_w = lr_w;
        this.lr_visible_b = lr_visible_b;
        this.lr_hidden_b = lr_hidden_b;
        this.reg_w = reg_w;
        this.momentumInitial = momentumInitial;
        this.momentumFinal = momentumFinal;
        this.modifier = modifier;
    }

    private void initialize() {

    }
    /**
     * Fits a separate RBM for each user, with 'tied' weights and biases
     * for the hidden and visible units.
     */
    public void fit(RbmOptions rbmOptions) throws IOException {

        int startAveraging = rbmOptions.maxepoch - rbmOptions.avglast;             

        ds.user_num = ds.matrix.getRows(); //    number of users
        ds.item_num = ds.matrix.getColumns();//  number of item
        
        _logger.info("got ratings from " + ds.user_num + " users for " + ds.item_num + " movies..");
        //Create batches
        //Batches batches = Utils.createBatches(user_num, rbmOptions.batchsize);
        ds.num_visible = ds.item_num;
        ds.num_hidden = rbmOptions.numhid;
 
        //initialize visible-hidden symmetric weights
        
        //1a. visual to hidden connection weights (1 per rating)
        ds.Wijk = new HashMap<>(5);

        for(int rating = 1; rating <= 5; rating++) {

            ds.Wijk.put(rating, DoubleMatrix.randn(ds.num_visible, ds.num_hidden).mul(0.1));
        }

        ds.Wijk_inc = new HashMap<>(5);
        for(int rating = 1; rating <= 5; rating++) {

            ds.Wijk_inc.put(rating, DoubleMatrix.zeros(ds.num_visible, ds.num_hidden));
        }

        //1b. gradients for visual to hidden connection weights (1 per rating)
        ds.pos_prods = new HashMap<>(5);
        for(int rating = 1; rating <= 5; rating++) {

            ds.pos_prods.put(rating, DoubleMatrix.zeros(ds.num_visible, ds.num_hidden));
        }
        
        ds.neg_prods = new HashMap<>(5);
        for(int rating = 1; rating <= 5; rating++) {

            ds.neg_prods.put(rating, DoubleMatrix.zeros(ds.num_visible, ds.num_hidden));
        }
        
        //2. biases for the hidden units
        ds.hidden_b = DoubleMatrix.zeros(1, ds.num_hidden);
        ds.hidbiasinc = DoubleMatrix.zeros(1, ds.num_hidden);

        //3. biases for the visible units (1 per rating)
        ds.visible_b = new HashMap<>(5);
        ds.visbiasesInc = new HashMap<>(5);
        for(int rating = 1; rating <= 5; rating++) {

            ds.visible_b.put(rating, DoubleMatrix.zeros(1, ds.num_visible));
            ds.visbiasesInc.put(rating, DoubleMatrix.zeros(1, ds.num_visible));
        }        

        //train for 'maxepoch' epochs
        for (int epoch = 1; epoch <= rbmOptions.maxepoch; epoch++) {

            _logger.info("Starting epoch " + (epoch + 1) + "\n");
            double err_sum = 0;
            
            // randomize the visiting order and then treat
            // each training case separately..
            List<Integer> visitingSeq = Utils.getSequence(0, ds.user_num - 1);
            Collections.shuffle(visitingSeq);
            
            for (int r = 0; r < ds.user_num; r++) {

                // each 'row' is in the form [0 0 5 4 2 0 0 3 ... 0 0 1 2 5],
                // If the value is > 0, it is the rating for that movie (column)
                // otherwise the rating is missing
                DoubleMatrix row = ds.matrix.getRow(visitingSeq.get(r));
                
                if(rbmOptions.debug)
                    _logger.info("Examining row.." + row.toString());
                
                // a row matrix (1 x num_visible) with 1's in the non-zero columns
                DoubleMatrix indicator = Utils.binaryMe(row);  
                if(rbmOptions.debug)
                    _logger.info("Indicator..." + indicator.toString());

                DoubleMatrix V = Utils.createRowMaskMatrix(row, 5);
                
                //positive phase
                DoubleMatrix poshidprobs = DoubleMatrix.zeros(1, ds.num_hidden);
                
                //add the biases
                poshidprobs.addi(ds.hidden_b);
                
                // the key is the 'rating' and and the value is the list of 
                // columns (movies) with this rating
                HashMap<Integer, List<Integer>> summarizeRatings = Utils.summarizeRatings(row);

                for (int rating = 1; rating <= 5; rating++) {

                    // 'columnList' contains all the column indices with
                    //rating equal to the current 'rating'
                    List<Integer> columnList = summarizeRatings.get(rating);
                    if (columnList == null) {
                        continue;
                    }

                    // 'rowMatrix' will have one's in the columns where the rating
                    // is equal to the current 'rating', and zero otherwise
                    DoubleMatrix rowMatrix = Utils.createRowMatrix(ds.num_visible, columnList);
                    DoubleMatrix wij = ds.Wijk.get(rating);
                    
                    // get the contribution from the active visible units
                    DoubleMatrix product = rowMatrix.mmul(wij);
                    poshidprobs.addi(product);
                }

                
                //take the logistic, to form probabilities for the hidden units
                poshidprobs = Utils.logistic(poshidprobs);
                
                
                // pos_prods = data' * poshidprobs
                for(int rating = 1; rating <=5; rating++) {  
                    DoubleMatrix posprod = ds.pos_prods.get(rating);
                    DoubleMatrix vRow = V.getRow(rating - 1);                    
                    posprod.addi(vRow.transpose().mmul(poshidprobs));
                    ds.pos_prods.put(rating, posprod);
                }
                
                
                if(rbmOptions.debug)
                    _logger.info("poshidprobs..." + poshidprobs.toString());
                
                //end of positive phase
                DoubleMatrix poshidstates = poshidprobs.ge(DoubleMatrix.rand(1, ds.num_hidden));
                if(rbmOptions.debug)
                    _logger.info("poshidstates..." + poshidstates.toString());                                                                
                
                    
                if(rbmOptions.debug)                
                    _logger.info("*** END OF POSITIVE PHASE \n\n\n");    
                
                
                
                //start negative phase        
                DoubleMatrix negdata = DoubleMatrix.zeros(5, ds.num_visible);
                
                for(int index = 0; index < indicator.getColumns(); index++ ) {
                    
                    // do not reconstruct missing ratings
                    if(indicator.get(0, index) == 0.0) { 
                        continue;
                    }
                    
                    
                    for(int rat = 0; rat < 5; rat++) {
                    
                        int rating = rat + 1;
                        
                        //get the bias for the specfic visible unit/rating
                        DoubleMatrix vbias = ds.visible_b.get(rating);
                        double bias = vbias.get(0, index);
                        
                        DoubleMatrix wij = ds.Wijk.get(rating);
                        
                        double sum = bias;
                        for(int hid = 0; hid < poshidstates.getColumns(); hid++) {
                            
                            //if the hidden is turned on, use it
                            if(poshidstates.get(0, hid) > 0.0) {
                             
                                sum += wij.get(index, hid);
                            }
                        }
                        
                        negdata.put(rat, index, sum);
                    }                                    
                }
                
                
                // zero negata values for the zero ratings
                negdata = Utils.softmax(negdata);
                
                
                DoubleMatrix neghidprobs = DoubleMatrix.zeros(1, ds.num_hidden);
                
                // add the biases for the hidden units
                neghidprobs.addi(ds.hidden_b);
                
                for(int index = 0; index < negdata.getColumns(); index++ ) {                    
                    
                    // if the rating is missing ignore it
                    if(negdata.getColumn(index).columnSums().get(0,0) == 0.0) {
                        continue;
                    }
                    
                    for (int k = 0; k < 5; k++) {
                    
                        int rating = k + 1;
                        DoubleMatrix wij = ds.Wijk.get(rating);
                        double visible_prob = negdata.get(k, index);
                        if(rbmOptions.debug)
                            _logger.info("visible prob = " + visible_prob);
                        DoubleMatrix wToHidden = wij.getRow(index);
                        if(rbmOptions.debug)
                            _logger.info("wToHidden is \n " + wToHidden.toString());
                        DoubleMatrix contributionToHiddenUnits = wToHidden.mul(visible_prob);   
                        if(rbmOptions.debug)
                            _logger.info("Adding.........." + contributionToHiddenUnits.toString());                        
                                                
                        neghidprobs.addi(contributionToHiddenUnits);
                     }
                }
                              
                neghidprobs = Utils.logistic(neghidprobs);
                if(rbmOptions.debug) {
                    _logger.info("neghidprobs.. " + neghidprobs.toString());
                }

                // neg_prods = negdata' * neghidprobs
                for(int rating = 1; rating <=5; rating++) {  
                    DoubleMatrix negprod = ds.neg_prods.get(rating);
                    DoubleMatrix vRow = negdata.getRow(rating - 1);                    
                    negprod.addi(vRow.transpose().mmul(neghidprobs));
                    ds.neg_prods.put(rating, negprod);
                }
                
                
                //the end for each user                
                DoubleMatrix error = V.sub(negdata);                 
                double err = error.norm2();
                
                //debug
                if(Double.isNaN(err)) {
                    _logger.info("Examining row.." + row.toString());
                    _logger.info("poshidprobs..." + poshidprobs.toString());
                    _logger.info("poshidstates..." + poshidstates.toString());                        
                    _logger.info("neghidprobs.. " + neghidprobs.toString());
                    _logger.info("error.. " + error.toString());
                } 
                else {
                    err_sum += err;
                }

                //set momentum
                double momentum = 0.0;
                if (epoch > startAveraging) {
                    momentum = momentumFinal;
                } else {
                    momentum = momentumInitial;
                }
            
                
                //calculate gradients
                ds.hidbiasinc = (ds.hidbiasinc.mul(momentum)).add((poshidprobs.sub(neghidprobs)).mul(lr_hidden_b *modifier/ds.user_num));
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix inc = ds.visbiasesInc.get(rating);
                    DoubleMatrix temp1 = inc.mul(momentum);
                    DoubleMatrix temp2 = (V.getRow(rating - 1).sub(negdata.getRow(rating - 1))).mul(lr_visible_b *modifier/ds.user_num);
                    inc = temp1.add(temp2);
                    ds.visbiasesInc.put(rating, inc);
                }
                
                             
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix inc = ds.Wijk_inc.get(rating);
                    DoubleMatrix temp1 = inc.mul(momentum);
                    DoubleMatrix temp2 = (ds.pos_prods.get(rating).sub(ds.neg_prods.get(rating))).mul(lr_w *modifier/ds.user_num);
                    DoubleMatrix temp3 = ds.Wijk.get(rating).mul(reg_w);
                    inc = temp1.add(temp2).sub(temp3);
                    ds.Wijk_inc.put(rating, inc);
                }
                                  
                
                                
                //update connection weights
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix wijk = ds.Wijk.get(rating);
                    ds.Wijk.put(rating, wijk.add(ds.Wijk_inc.get(rating)));
                }
                
                ds.hidden_b = ds.hidden_b.add(ds.hidbiasinc);
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix vis = ds.visible_b.get(rating);
                    ds.visible_b.put(rating, vis.add(ds.visbiasesInc.get(rating)));
                }
                    
            }
            
                      
            // reset 'Winj_inc' matrix 
            for(int rating = 1; rating <=5; rating++ ) {
                    ds.Wijk_inc.put(rating, DoubleMatrix.zeros(ds.num_visible, ds.num_hidden));
            }
            
            _logger.info("Epoch " + epoch + " error " + err_sum + "\n");
            
        } // end of epoch
                          
    }
                       
    
    /**
     * Reads the file with the user-item ratings. The expected format is 
     * 'user'<tab/>'item'<tab/>'rating
     * @param file 
     */
    public void loadRatings(String file) {
               
        HashMap<String, List<Rating>> ratingsMap = new HashMap<>(10000);
       
        try {
            BigFile f = new BigFile(file);
            HashSet<String> items = new HashSet<String>(5000);

            Iterator<String> iterator = f.iterator();
            while (iterator.hasNext()) {
                String line = iterator.next();
                String[] splits = line.split("\t");

                String userId = splits[0];
                String itemId = splits[1];
                double rating = Double.parseDouble(splits[2]);

                List<Rating> ratingsList = ratingsMap.get(userId);
                if(ratingsList == null) {
                    ratingsList = new ArrayList<>();
                }

                ratingsList.add(new Rating(itemId, rating));
                ratingsMap.put(userId, ratingsList);
                items.add(itemId);
            }

            System.out.println("Found " + ratingsMap.keySet().size() + " users " +
                    " and " + items.size() + " items..");
            
            // initialize all with zeros
            ds.matrix = DoubleMatrix.zeros(ratingsMap.keySet().size(), items.size());
            
            ds.user2Index = new HashMap<>(ratingsMap.keySet().size());
            ds.index2User = new HashMap<>(ratingsMap.keySet().size());
            
            
            int rowIndex = 0;
            for(Map.Entry<String, List<Rating>> entry : ratingsMap.entrySet()) {
                
                String userId = entry.getKey();
                ds.user2Index.put(userId, rowIndex);
                ds.index2User.put(rowIndex, userId);
                rowIndex++;
            }
            int usersNr = rowIndex;
            
            
            int columnIndex = 0;
            ds.item2Index = new HashMap<>(items.size());
            for(String item : items) {
                
                ds.item2Index.put(item, columnIndex);
                columnIndex++;
            }
                        
            for(int row = 0; row < usersNr; row++) {
                
                String userId = ds.index2User.get(row);
                List<Rating> ratings = ratingsMap.get(userId);
                for(Rating rating : ratings) {
                    
                    String item = rating.itemId;
                    double val = rating.rating;
                    
                    ds.matrix.put(row, ds.item2Index.get(item), val);
                }                
            }
                  
                         
        } catch (Exception ex) {

            ex.printStackTrace();
        }
        
    }

    public double predict(String userId, String itemId, PredictionType predictionType) {

        int userIndex = ds.user2Index.get(userId);
        int itemIndex = ds.item2Index.get(itemId);

        DoubleMatrix user_ratings_so_far = ds.matrix.getRow(userIndex);
                
        //positive phase
        DoubleMatrix poshidprobs = DoubleMatrix.zeros(1, ds.num_hidden);

        //add the biases
        poshidprobs.addi(ds.hidden_b);
                   
        // the key is the 'rating' and and the value is the list of 
        // indices that contain that rating
        HashMap<Integer, List<Integer>> summarize = Utils.summarizeRatings(user_ratings_so_far);

        for (int rating = 1; rating <= 5; rating++) {

            // 'columnList' contains all the column indices with
            //rating equal to the current 'rating'
            List<Integer> columnList = summarize.get(rating);
            if (columnList == null) {
                continue;
            }

            // 'rowMatrix' will have one's in the columns where the rating
            // is equal to the current 'rating', and zero otherwise
            DoubleMatrix rowMatrix = Utils.createRowMatrix(ds.num_visible, columnList);
            DoubleMatrix wij = ds.Wijk.get(rating);

            //get the contribution from the active visible units
            DoubleMatrix product = rowMatrix.mmul(wij);
            poshidprobs.addi(product);
        }

        //take the logistic, to form probabilities for the hidden units
        poshidprobs = Utils.logistic(poshidprobs);
        DoubleMatrix poshidstates = poshidprobs.ge(DoubleMatrix.rand(1, ds.num_hidden));
        
        DoubleMatrix negdata = DoubleMatrix.zeros(5, 1);

        for (int rat = 0; rat < 5; rat++) {

            int rating = rat + 1;

            //get the bias for the specfic visible unit/rating
            DoubleMatrix vbias = ds.visible_b.get(rating);
            double bias = vbias.get(0, itemIndex);

            DoubleMatrix wij = ds.Wijk.get(rating);

            double sum = bias;
            for (int hid = 0; hid < poshidstates.getColumns(); hid++) {

                //if the hidden is turned on, use it
                if (poshidstates.get(0, hid) > 0.0) {

                    sum += wij.get(itemIndex, hid);
                }
            }

            negdata.put(rat, 0, sum);
        }
        
        negdata = Utils.softmax(negdata);
        
        if(predictionType.equals(PredictionType.MAX)) {
            
            int max_index = 0;
            double max_value = negdata.get(0,0);
            
            for(int i = 1; i < negdata.getRows(); i++ ) {
                double current = negdata.get(i,0);
                if(current > max_value) {
                    max_index = i;
                    max_value = current;
                }
            }
            
            return (max_index + 1)*1.0;
                       
        }else if(predictionType.equals(PredictionType.MEAN)) {
            
            double mean = 0.0;
                       
            for(int i = 0; i < negdata.getRows(); i++ ) {

                mean += negdata.get(i,0) * (i + 1);
                
            }
            
            return mean;
            
            
        }
        
        return 0.0;
    }
                
                   
}
