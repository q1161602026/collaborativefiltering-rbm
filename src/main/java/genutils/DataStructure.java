package genutils;

import org.jblas.DoubleMatrix;

import java.util.HashMap;

/**
 * Created by mjh on 2018/1/30.
 */
public class DataStructure {
    public HashMap<Integer, DoubleMatrix> Wijk;
    public DoubleMatrix hidden_b;
    public HashMap<Integer, DoubleMatrix> visible_b;

    public DoubleMatrix matrix;

    public int user_num; //    number of users
    public int item_num;//  number of item

    public HashMap<String, Integer> item2Index;
    public HashMap<String, Integer> user2Index;
    public HashMap<Integer, String> index2User;

    public int num_hidden; // number of units in the hidden layer
    public int num_visible; // number of visible units

    public HashMap<Integer, DoubleMatrix> Wijk_inc;
    public HashMap<Integer, DoubleMatrix> pos_prods;
    public HashMap<Integer, DoubleMatrix> neg_prods;
    public HashMap<Integer, DoubleMatrix> visbiasesInc;
    public DoubleMatrix hidbiasinc ;




}
