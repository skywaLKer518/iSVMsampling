public class Environment{
	static final int trainSize = 100;
	static final int dataSetSize = 10000;
	static final int testSize = dataSetSize - trainSize;
	static final int dataSetNum = 1;//50;
	static final int maxComponent = trainSize;
	static final int dataCateNum = 2;
	static final double reduce = 30;
	static final int Times = 200;//00;
	
	public static double sampleStandardNormalUnivariate(){ // return a sample from standard normal distribution
		double r1 = 1,r2 = 1;
		r1 = Math.random();r2 = Math.random();
		return Math.sqrt(-2 * Math.log(r1)) * Math.sin(2 * Math.PI * r2);
	}	
	
	public static double sampleNormalUnivariate(double mean, double var ){ // variance = var 
		double a = sampleStandardNormalUnivariate();
		return Math.sqrt(var) * a + mean;
	}
	
	Environment(){
	}
}
