public class Data{
	private int setting = 1;
	
	
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
	
	public static double sampleNormalUnivariate(double mean, double var ){ // variance = var = sigma * sigma 
		double a = sampleStandardNormalUnivariate();
		return Math.sqrt(var) * a + mean;
	}
	
	
	
	void newData(){

	}

}