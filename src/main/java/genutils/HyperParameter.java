package genutils;

/**
 * Created by TonyMar on 2018/1/30.
 */
public class HyperParameter {
    public HyperParameter(double lr_w, double lr_visible_b, double lr_hidden_b,
                          double reg_w, double momentumInitial, double momentumFinal,
                          double modifier) {

        this.lr_w = lr_w;
        this.lr_visible_b = lr_visible_b;
        this.lr_hidden_b = lr_hidden_b;
        this.reg_w = reg_w;
        this.momentumInitial = momentumInitial;
        this.momentumFinal = momentumFinal;
        this.modifier = modifier;

    }

    public double lr_w;
    public double lr_visible_b;
    public double lr_hidden_b;
    public double reg_w;
    public double momentumInitial;
    public double momentumFinal;
    public double modifier;
}
