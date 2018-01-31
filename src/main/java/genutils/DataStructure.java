package genutils;

import org.jblas.DoubleMatrix;

import java.util.HashMap;

/**
 * Created by TonyMar on 2018/1/30.
 */
public class DataStructure {

    public int user_num; //  number of users
    public int item_num; //  number of item

    public int num_hidden;  // number of units in the hidden layer
    public int num_visible; // number of visible units

    public HashMap<String, Integer> item2Index;
    public HashMap<String, Integer> user2Index;
    public HashMap<Integer, String> index2User;

    public DoubleMatrix matrix;
    public DoubleMatrix matrix_train;
    public DoubleMatrix matrix_test;

    public HashMap<Integer, DoubleMatrix> Wijk;
    public DoubleMatrix hidden_b;
    public HashMap<Integer, DoubleMatrix> visible_b;

    public HashMap<Integer, DoubleMatrix> Wijk_inc;
    public DoubleMatrix hidden_b_inc;
    public HashMap<Integer, DoubleMatrix> visible_b_inc;

    public DoubleMatrix pos_hid_probs;
    public DoubleMatrix neg_hid_probs;

    public DoubleMatrix pos_hid_states;

    public HashMap<Integer, DoubleMatrix> pos_prods; // <vi,hj> k data
    public HashMap<Integer, DoubleMatrix> neg_prods; // <vi,hj> k model


}
