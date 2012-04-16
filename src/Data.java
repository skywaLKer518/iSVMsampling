public class Data{
	public static double sampleStandardNormalUnivariate(){ // return a sample from standard normal distribution
		double r1 = 1,r2 = 1;
		r1 = Math.random();r2 = Math.random();
		return Math.sqrt(-2 * Math.log(r1)) * Math.sin(2 * Math.PI * r2);
	}
	
	public static double sampleNormalUnivariate(double mean, double var ){ // variance = var = sigma * sigma 
		double a = sampleStandardNormalUnivariate();
		return Math.sqrt(var) * a + mean;
	}

	void newData(){
	}
	void clearData(){
	}

}