/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package drivers;

import deeplearning.tools.rbmforcollaborativefiltering.CollaborativeFilteringRBM;
import genutils.HyperParameter;
import genutils.RbmOptions;
import java.io.IOException;
import java.util.logging.Logger;

/**
 *
 * @author thanos
 */
public class TestCollaborativeFilteringRBM {

    private static final Logger _logger = Logger.getLogger(TestCollaborativeFilteringRBM.class.getName());

    public static void main(String[] args) throws IOException {

        _logger.info("Loading data..");
        HyperParameter hp = new HyperParameter(0.1, 0.2, 0.1, 0.002, 0.9, 20.0);

        RbmOptions ro = new RbmOptions();
        ro.cdk = 5;
        ro.epoch = 10;
        ro.evaluateEvery = 2;
        ro.num_hidden = 100;
        ro.debug = false;

        //CollaborativeFilteringLayer fit = CollaborativeFilteringRBM.fit(data, ro);
        CollaborativeFilteringRBM rbmCF = new CollaborativeFilteringRBM(hp,ro);
        rbmCF.loadRatings("./data/" + "u.data",0.2);
        rbmCF.fit();

    }

}
