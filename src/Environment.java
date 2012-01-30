public class Environment{
	static final int trainSize = 100;
	static final int dataSetSize = 10000;
	static final int dataSetNum = 50;
	static final int maxComponent = 100;
	
	
	public static double sampleStandardNormalUnivariate(){ // return a sample from standard normal distribution
		double r1 = 1,r2 = 1;
		double tmp = r1*r1 + r2*r2;
		while (tmp > 1){
			r1 = 2 * Math.random() - 1;
			r2 = 2 * Math.random() - 1;
			tmp = r1*r1 + r2*r2;
		}
		return r1 * Math.sqrt(-2 * Math.log(Math.abs(r1)) / tmp);
	}	
	
	public static double sampleNormalUnivariate(double mean, double var ){ // variance = var 
		double a = sampleStandardNormalUnivariate();
		return Math.sqrt(var) * a + mean;
	}
	
	
	
	
	static final double Coefficient = (1 / Math.sqrt(2 * Math.PI));  
	static final int MaxStateNumber = 100;
	static final int MaxCateNumber = 100;
	static final int Times = 2000;
	static final int R = 5;  // repeated times
	static final double delta = 0.02;
	static final double Error = 999;
	static final double F_sigma = 0.1;
	private static double x[] = new double[501];  // used to get sample area
	private static double prior[] = new double[501];  // used to store probability for each area based on G_0
	private static double posterior[] = new double[501];
	static final double thetaMax = -5;
	static final double thetaMin = -5;
	static final double Expand = 10000000000.0; // make sure the normalization factor is not too large...TODO
	
	static int postErrorTimes = 0;
	Environment(){
		// x[0] = -5, x[500] = 5;
		x[0] = -5;
		for (int i = 1; i < 501; i ++){
			x[i] = x[i-1]+ 0.02;
		}
		// construct a distribution of G_0
		double sum = 0;
		for (int i = 0; i < 501; i ++){
			prior[i] = getNormalP(x[i],DP.u,DP.sigma) * 0.02;
			sum += prior[i];
		}
		System.out.println("sum = " +sum);
	}
	private double getNormalP(double a, double mean, double sigma){
		double p = 1;
		p = 1 / Math.sqrt(2 * Math.PI) / sigma;
		p = p * Math.pow(Math.E, -0.5 * (a - mean) * (a - mean) / sigma / sigma);
		return p;
	}
	public static int AreaFromNormal(){
		double r = Math.random();
		double F = 0;
		for (int i = 0; i < 501; i ++){
			F += prior[i];
			if (r <= F){
				return i;
			}
		}
		return -1; // error
	}
	public static double sampleStandardNormal(){
		int area = AreaFromNormal();
		if (area == -1){
			System.out.println("Error in post sampling");
			return Error;
		}
		// now uniformly sample from [-5 + 0.02i - 0.01, -5 + 0.02i +0.01]
		double r = Math.random();
		return (-5 + 0.02 * area - 0.01 + r * 0.02);
	}
	
	// given all data points, and number, return post theta
	public static double drawPosterior(double[] d, int size) {
		double sumtmp = 0;
		double postSum = 0;
		for (int i = 0; i < 501; i++){
			posterior[i] = prior[i];
			for (int j = 0; j < size; j ++){
//				System.out.println("en "+F(x[i],0.1,d[j]));
				posterior[i] *= F(x[i],0.1,d[j]) * Expand;
			}
			sumtmp += posterior[i];
		}
if (sumtmp == 0){
			System.out.println("here print post");
			for (int j = 0; j < size; j ++){
				System.out.print("data["+j+"] = "+d[j]+";  ");
			}
			System.out.println();
		}
		
		for (int i = 0; i < 501; i ++){
			posterior[i] /= sumtmp;
			postSum+= posterior[i];
		}
//		System.out.println("postSum =: "+ postSum);
		// sample from posterior
		double r = Math.random();
		double F = 0;
		int pos = -1;
		for (int i = 0; i < 501; i ++){
			F += posterior[i];
			if (r <= F){
				pos = i;
				break;
			}
		}
		if (pos == -1){
			postErrorTimes++;
			System.out.println("sumtmp: "+sumtmp);
			System.out.println("Error in post sampling");
//			System.exit(-1);
			return 0;
		}
		double r1 = Math.random();
		return (-5 + 0.02 * pos - 0.01 + r1 * 0.02);
		
	}
	
	private static double F(double theta,double sigma, double y){
		return  (Coefficient / sigma) * Math.pow(Math.E, -0.5 * (y - theta) * (y - theta) / sigma / sigma);
	}
}