/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package genutils;


/**
 *
 * @author Thanos
 */
public class RbmOptions {

    public int cdk = 1;
    public int epoch = 10;
    public int avglast = 0;
    public double penalty = 2e-4;
    public boolean verbose = true;
    public boolean anneal = false;
    public int num_hidden = 500;
    public boolean debug = false;
    public boolean restart = true;
    public int createSnapshotEvery = 100;

}
