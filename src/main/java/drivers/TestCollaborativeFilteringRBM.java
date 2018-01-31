/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package drivers;

import deeplearning.tools.rbmforcollaborativefiltering.CollaborativeFilteringRBM;
import deeplearning.tools.rbmforcollaborativefiltering.PredictionType;
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
        HyperParameter hp = new HyperParameter(0.1, 0.2, 0.1, 0.002, 0.5, 0.9, 20.0);

        RbmOptions ro = new RbmOptions();
        ro.maxepoch = 3;
        ro.avglast = 5;
        ro.num_hidden = 100;
        ro.debug = false;

        //CollaborativeFilteringLayer fit = CollaborativeFilteringRBM.fit(data, ro);
        CollaborativeFilteringRBM rbmCF = new CollaborativeFilteringRBM(hp,ro);
        rbmCF.loadRatings("./data/" + "u.data",0.2);
        rbmCF.fit();
        System.out.println("evaluating ......");
        rbmCF.evaluate();

        System.out.println("Mean prediction = " + rbmCF.predict("166", "346", PredictionType.MEAN));

    }

}
