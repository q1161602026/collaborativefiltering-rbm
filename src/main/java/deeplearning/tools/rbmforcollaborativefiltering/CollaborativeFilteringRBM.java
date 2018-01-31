/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package deeplearning.tools.rbmforcollaborativefiltering;

import genutils.*;

import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;
import org.jblas.DoubleMatrix;

/**
 *
 * @author thanos
 */
public class CollaborativeFilteringRBM {


    private static final java.util.logging.Logger _logger = Logger.getLogger(CollaborativeFilteringRBM.class.getName());

    private RbmOptions ro;

    private HyperParameter hp;

    private DataStructure ds;

    private TempDataStructure tds;



    public CollaborativeFilteringRBM(HyperParameter hp, RbmOptions ro) {
        this.hp = hp;
        this.ro = ro;

        ds = new DataStructure();
        tds = new TempDataStructure();
    }

    private void initialize() throws IOException {

        ds.user_num = ds.matrix.getRows(); //    number of users
        ds.item_num = ds.matrix.getColumns();//  number of item

        _logger.info("got ratings from " + ds.user_num + " users for " + ds.item_num + " movies..");
        //Create batches
        //Batches batches = Utils.createBatches(user_num, ro.batchsize);
        ds.num_visible = ds.item_num;
        ds.num_hidden = ro.num_hidden;

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
        ds.hidden_b_inc = DoubleMatrix.zeros(1, ds.num_hidden);

        //3. biases for the visible units (1 per rating)
        ds.visible_b = new HashMap<>(5);
        ds.visible_b_inc = new HashMap<>(5);
        for(int rating = 1; rating <= 5; rating++) {

            ds.visible_b.put(rating, DoubleMatrix.zeros(1, ds.num_visible));
            ds.visible_b_inc.put(rating, DoubleMatrix.zeros(1, ds.num_visible));
        }

    }
    private boolean prepare(DoubleMatrix matrix, int r) {

        tds.row = matrix.getRow(r);

        if(ro.debug)
            _logger.info("Examining row.." + tds.row.toString());

        // a row matrix (1 x num_visible) with 1's in the non-zero columns
        tds.binaryMatrix = Utils.getBinaryMatrix(tds.row);

        DoubleMatrix zeroAll = DoubleMatrix.zeros(tds.row.getRows(), tds.row.getColumns());

        if (tds.binaryMatrix.equals(zeroAll)){
            return false;
        }

        if(ro.debug)
            _logger.info("Indicator..." + tds.binaryMatrix.toString());

        tds.oneHotEncoder = Utils.getOneHotEncoder(tds.row, 5);
        return true;

    }

    private void positive_phase(DataStructure ds,  TempDataStructure tds) {
        //positive phase
        ds.pos_hid_probs = DoubleMatrix.zeros(1, ds.num_hidden);

        //add the biases
        ds.pos_hid_probs.addi(ds.hidden_b);

        // the key is the 'rating' and and the value is the list of
        // columns (movies) with this rating
        HashMap<Integer, List<Integer>> summarizeRatings = Utils.summarizeRatings(tds.row);

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
            ds.pos_hid_probs.addi(product);
        }

        //take the logistic, to form probabilities for the hidden units
        ds.pos_hid_probs = Utils.logistic(ds.pos_hid_probs);

        // pos_prods = data' * ds.pos_hid_probs
        for(int rating = 1; rating <= 5; rating++) {
            DoubleMatrix pos_prod = ds.pos_prods.get(rating);
            DoubleMatrix vRow = tds.oneHotEncoder.getRow(rating - 1);
            pos_prod.addi(vRow.transpose().mmul(ds.pos_hid_probs));
            ds.pos_prods.put(rating, pos_prod);
        }

        if(ro.debug)
            _logger.info("ds.pos_hid_probs..." + ds.pos_hid_probs.toString());

        //end of positive phase
        ds.pos_hid_states = ds.pos_hid_probs.ge(DoubleMatrix.rand(1, ds.num_hidden));
        if(ro.debug)
            _logger.info("poshidstates..." + ds.pos_hid_states.toString());


        if(ro.debug)
            _logger.info("*** END OF POSITIVE PHASE \n\n\n");

    }

    private DoubleMatrix reconstructEncoder(DataStructure ds, TempDataStructure tds){
        DoubleMatrix reconstructEncoder = DoubleMatrix.zeros(5, ds.num_visible);

        for(int itemIndex = 0; itemIndex < tds.binaryMatrix.getColumns(); itemIndex++ ) {

            // do not reconstruct missing ratings
            if(tds.binaryMatrix.get(0, itemIndex) == 0.0) {
                continue;
            }

            for(int rating = 1; rating <= 5; rating++) {

                //get the bias for the specfic visible unit/rating
                DoubleMatrix visible_b = ds.visible_b.get(rating);
                double bias = visible_b.get(0, itemIndex);

                DoubleMatrix wij = ds.Wijk.get(rating);

                double sum = bias;
                for(int hid = 0; hid < ds.pos_hid_states.getColumns(); hid++) {

                    //if the hidden is turned on, use it
                    if(ds.pos_hid_states.get(0, hid) > 0.0) {

                        sum += wij.get(itemIndex, hid);
                    }
                }

                reconstructEncoder.put(rating - 1, itemIndex, sum);
            }
        }
        // zero negata values for the zero ratings
        reconstructEncoder = Utils.softmax(reconstructEncoder);
        return reconstructEncoder;
    }

    private void negative_phase(DataStructure ds, TempDataStructure tds) {

        //start negative phase
        tds.reconstructEncoder = reconstructEncoder(ds, tds);

        ds.neg_hid_probs = DoubleMatrix.zeros(1, ds.num_hidden);

        // add the biases for the hidden units
        ds.neg_hid_probs.addi(ds.hidden_b);

        for(int itemIndex = 0; itemIndex < tds.reconstructEncoder.getColumns(); itemIndex++ ) {

            // if the rating is missing ignore it
            if(tds.binaryMatrix.get(0, itemIndex) == 0.0) {
                continue;
            }

            for (int k = 0; k < 5; k++) {

                int rating = k + 1;
                DoubleMatrix wij = ds.Wijk.get(rating);
                double visible_prob = tds.reconstructEncoder.get(k, itemIndex);
                if(ro.debug)
                    _logger.info("visible prob = " + visible_prob);
                DoubleMatrix witemj = wij.getRow(itemIndex);
                if(ro.debug)
                    _logger.info("witemToj is \n " + witemj.toString());
                DoubleMatrix contributionToHiddenUnits = witemj.mul(visible_prob);
                if(ro.debug)
                    _logger.info("Adding.........." + contributionToHiddenUnits.toString());

                ds.neg_hid_probs.addi(contributionToHiddenUnits);
            }
        }

        ds.neg_hid_probs = Utils.logistic(ds.neg_hid_probs);
        if(ro.debug) {
            _logger.info("neg_hid_probs.. " + ds.neg_hid_probs.toString());
        }

        // neg_prods = reconstructEncoder' * neg_hid_probs
        for(int rating = 1; rating <= 5; rating++) {
            DoubleMatrix neg_prod = ds.neg_prods.get(rating);
            DoubleMatrix vRow = tds.reconstructEncoder.getRow(rating - 1);
            neg_prod.addi(vRow.transpose().mmul(ds.neg_hid_probs));
            ds.neg_prods.put(rating, neg_prod);
        }
    }

    private void update_weight(int epoch) {
        int startAveraging = ro.maxepoch - ro.avglast;

        //set momentum_coeff
        double momentum_coeff;
        if (epoch > startAveraging) {
            momentum_coeff = hp.momentum_coefficient_final;
        } else {
            momentum_coeff = hp.momentum_coefficient_initial;
        }

        //calculate gradients
        ds.hidden_b_inc = (ds.hidden_b_inc.mul(momentum_coeff)).add((ds.pos_hid_probs.sub(ds.neg_hid_probs)).mul(hp.lr_hidden_b * hp.modifier / ds.user_num));

        for(int rating = 1; rating <= 5; rating++ ) {
            DoubleMatrix inc = ds.visible_b_inc.get(rating);
            DoubleMatrix temp1 = inc.mul(momentum_coeff);
            DoubleMatrix temp2 = (tds.oneHotEncoder.getRow(rating - 1).sub(tds.reconstructEncoder.getRow(rating - 1))).mul(hp.lr_visible_b * hp.modifier/ ds.user_num);
            inc = temp1.add(temp2);
            ds.visible_b_inc.put(rating, inc);
        }

        for(int rating = 1; rating <= 5; rating++ ) {
            DoubleMatrix inc = ds.Wijk_inc.get(rating);
            DoubleMatrix temp1 = inc.mul(momentum_coeff);
            DoubleMatrix temp2 = (ds.pos_prods.get(rating).sub(ds.neg_prods.get(rating))).mul(hp.lr_w * hp.modifier/ ds.user_num);
            DoubleMatrix temp3 = ds.Wijk.get(rating).mul(hp.reg_w);
            inc = temp1.add(temp2).sub(temp3);
            ds.Wijk_inc.put(rating, inc);
        }

        //update connection weights

        ds.hidden_b = ds.hidden_b.add(ds.hidden_b_inc);

        for(int rating = 1; rating <= 5; rating++ ) {
            DoubleMatrix vis = ds.visible_b.get(rating);
            ds.visible_b.put(rating, vis.add(ds.visible_b_inc.get(rating)));
        }

        for(int rating = 1; rating <= 5; rating++ ) {
            DoubleMatrix wijk = ds.Wijk.get(rating);
            ds.Wijk.put(rating, wijk.add(ds.Wijk_inc.get(rating)));
        }

    }

    /**
     * Fits a separate RBM for each user, with 'tied' weights and biases
     * for the hidden and visible units.
     */
    public void fit() throws IOException {

        initialize();

        List<Integer> userRandomIndex = Utils.getSequence(0, ds.user_num - 1);

        //train for 'maxepoch' epochs
        for (int epoch = 1; epoch <= ro.maxepoch; epoch++) {

            _logger.info("Starting epoch " + (epoch + 1) + "\n");
            double err_sum = 0;

            // randomize the visiting order and then treat
            // each training case separately..
            Collections.shuffle(userRandomIndex);

            for (int r = 0; r < userRandomIndex.size(); r++) {

                // each 'row' is in the form [0 0 5 4 2 0 0 3 ... 0 0 1 2 5],
                // If the value is > 0, it is the rating for that movie (column)
                // otherwise the rating is missing

                boolean prepared = prepare(ds.matrix_train, userRandomIndex.get(r));

                if (!prepared){
                    continue;
                }

                positive_phase(ds, tds);

                negative_phase(ds, tds);

                update_weight(epoch);

                //the end for each user
                DoubleMatrix error = tds.oneHotEncoder.sub(tds.reconstructEncoder);
                double err = error.norm2();

                //debug
                if(Double.isNaN(err)) {
                    _logger.info("Examining row.." + tds.row.toString());
                    _logger.info("pos_hid_probs..." + ds.pos_hid_probs.toString());
                    _logger.info("poshidstates..." + ds.pos_hid_states.toString());
                    _logger.info("neg_hid_probs.. " + ds.neg_hid_probs.toString());
                    _logger.info("error.. " + error.toString());
                }
                else {
                    err_sum += err;
                }

            }

            // reset 'Winj_inc' matrix
            for(int rating = 1; rating <= 5; rating++ ) {
                ds.Wijk_inc.put(rating, DoubleMatrix.zeros(ds.num_visible, ds.num_hidden));
            }

            _logger.info("Epoch " + epoch + " error " + err_sum + "\n");

        } // end of epoch

    }

    public void evaluate() throws IOException {

        List<Integer> userIndex = Utils.getSequence(0, ds.user_num - 1);

        //train for 'maxepoch' epochs

        double err_sum = 0;
        int count = 0;

        // randomize the visiting order and then treat
        // each training case separately..

        for (int r = 0; r < userIndex.size(); r++) {

            boolean prepared = prepare(ds.matrix_train, userIndex.get(r));

            if (!prepared) {
                continue;
            }

            positive_phase(ds, tds);

            prepared = prepare(ds.matrix_test, userIndex.get(r));

            if (!prepared) {
                continue;
            }

            tds.reconstructEncoder = reconstructEncoder(ds, tds);

            //the end for each user
            int[] true_list = tds.oneHotEncoder.columnArgmaxs();
            int[] predict_list = tds.reconstructEncoder.columnArgmaxs();
            double err_sub = 0;
            for (int i = 0; i < true_list.length; i++) {
                if (true_list[i] == 0) {
                    continue;
                }
                double err = true_list[i] - predict_list[i];
                err_sub += err *  err;
                count++;
            }

            //debug
            if(Double.isNaN(err_sub)) {
                _logger.info("Examining row.." + tds.row.toString());
                _logger.info("pos_hid_probs..." + ds.pos_hid_probs.toString());
                _logger.info("poshidstates..." + ds.pos_hid_states.toString());
                _logger.info("neg_hid_probs.. " + ds.neg_hid_probs.toString());
//                _logger.info("error.. " + error.toString());
            }
            else {
                err_sum += err_sub;
            }
        }
        System.out.println("RMSE:" + Math.sqrt(err_sum / count));

    }

    /**
     * Reads the file with the user-item ratings. The expected format is
     * 'user'<tab/>'item'<tab/>'rating
     * @param file
     */
    public void loadRatings(String file, double validation_split) {

        HashMap<String, List<Rating>> ratingsMap = new HashMap<>(10000);

        try {
            BigFile f = new BigFile(file);
            HashSet<String> items = new HashSet<>(5000);

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
            int usersNum = rowIndex;


            int columnIndex = 0;
            ds.item2Index = new HashMap<>(items.size());
            for(String item : items) {

                ds.item2Index.put(item, columnIndex);
                columnIndex++;
            }

            for(int user = 0; user < usersNum; user++) {

                String userId = ds.index2User.get(user);
                List<Rating> ratings = ratingsMap.get(userId);
                for(Rating rating : ratings) {

                    String item = rating.itemId;
                    double val = rating.rating;

                    ds.matrix.put(user, ds.item2Index.get(item), val);
                }
            }
            if (validation_split > 0.0) {
                DoubleMatrix mask = DoubleMatrix.rand(ratingsMap.keySet().size(), items.size());
                DoubleMatrix train_mask = mask.ge(validation_split);
                DoubleMatrix test_mask = mask.le(validation_split);
                ds.matrix_train = ds.matrix.mul(train_mask);
                ds.matrix_test = ds.matrix.mul(test_mask);
            }
            else {
                ds.matrix_train = ds.matrix;
            }

            f.Close();

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
