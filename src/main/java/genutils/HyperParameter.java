package genutils;

/**
 * Created by TonyMar on 2018/1/30.
 */
public class HyperParameter {
    public HyperParameter(double lr_w, double lr_visible_b, double lr_hidden_b,
                          double reg_w, double momentum_coefficient_initial, double momentum_coefficient_final,
                          double modifier) {

        this.lr_w = lr_w;
        this.lr_visible_b = lr_visible_b;
        this.lr_hidden_b = lr_hidden_b;
        this.reg_w = reg_w;
        this.momentum_coefficient_initial = momentum_coefficient_initial;
        this.momentum_coefficient_final = momentum_coefficient_final;
        this.modifier = modifier;

    }

    public double lr_w;
    public double lr_visible_b;
    public double lr_hidden_b;
    public double reg_w;
    public double momentum_coefficient_initial;
    public double momentum_coefficient_final;
    public double modifier;
}
