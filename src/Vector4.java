/*
 * setting 1
 * u1 ~ N(1,1),u2 ~ N(2,1)
 * log(sigma^2) ~ N(0,2^2)
 * 
 */
class Vector4 extends Data{
	private double v1[],v2[],v3[],v4[];
	private int lable[];
	
	// test TODO
	private double F = 0;
	private double num = 0;
	Vector4(){
		v1 = new double[Environment.dataSetSize];
		v2 = new double[Environment.dataSetSize];
		v3 = new double[Environment.dataSetSize];
		v4 = new double[Environment.dataSetSize];
		lable = new int[Environment.dataSetSize];
	}
	
	void newData(){
		for(int i = 0; i < Environment.dataSetSize / 2; i++){
			v1[i] = sampleNormalUnivariate(1,1);
			v2[i] = sampleNormalUnivariate(1,1);
			v3[i] = sampleNormalUnivariate(1,1);
			v4[i] = sampleNormalUnivariate(1,1);
		}
		for(int i = Environment.dataSetSize / 2; i <Environment.dataSetSize; i++){
			v1[i] = sampleNormalUnivariate(2,1);
			v2[i] = sampleNormalUnivariate(2,1);
			v3[i] = sampleNormalUnivariate(2,1);
			v4[i] = sampleNormalUnivariate(2,1);
		}
		for (int i = 0; i < Environment.dataSetSize; i++){
			v1[i] *= Math.sqrt(Math.exp(sampleNormalUnivariate(0,2*2)));
			v2[i] *= Math.sqrt(Math.exp(sampleNormalUnivariate(0,2*2)));
			v3[i] *= Math.sqrt(Math.exp(sampleNormalUnivariate(0,2*2)));
			v4[i] *= Math.sqrt(Math.exp(sampleNormalUnivariate(0,2*2)));
			lable[i] = Lable(i);
		}
		System.out.println("Data generation completed!");
	}

	private int Lable(int i) {
		double a,b,c,theta,p;
		a = sampleNormalUnivariate(1,0.5*0.5);
		b = sampleNormalUnivariate(1,0.5*0.5);
		c = sampleNormalUnivariate(1,0.5*0.5);
		theta = - a * Math.sin(v1[i] * v1[i] * v1[i] + 1.2) -v1[i] * Math.cos(b * v2[i] + 0.7) - c * v3[i] + 2;
		p = 1 / (1 + Math.exp(-theta));
		if (Math.random() < p)
			return 1;
		else
			return 0;
	}
	
//	// sum of F,
//	public double getSumF(int[] da, int size) {
//		double r;
//		for (int i = 0; i < size; i ++){
//			r += disFunc();
//		}
//		return 0;
//	}
	
	public double disFunc(double[] eta,int i){ // in train
		if (lable[i] == 0){// then f(y,x) = (x[1],x[2],x[3],x[4],0,0,0,0)
			return (eta[0] * v1[i] + eta[1] * v2[i] + eta[2] * v3[i] + eta[3] * v4[i]);
		}
		else{ // then f(y,x) = (0,0,0,0,x[1],x[2],x[3],x[4])
			return (eta[4] * v1[i] + eta[5] * v2[i] + eta[6] * v3[i] + eta[7] * v4[i]);
		}
	}
	
	public double disFunc(double[] eta,int i,int y){ // in test
		double f = 0;
		if (y == 0){// then f(y,x) = (x[1],x[2],x[3],x[4],0,0,0,0)
			f =(eta[0] * v1[i] + eta[1] * v2[i] + eta[2] * v3[i] + eta[3] * v4[i]);
//			printV(i);
//			System.out.println("F = "+f);
			F += f;
			num ++;
			return f;
		}
		else{ // then f(y,x) = (0,0,0,0,x[1],x[2],x[3],x[4])
			f = (eta[4] * v1[i] + eta[5] * v2[i] + eta[6] * v3[i] + eta[7] * v4[i]);
//			printV(i);
//			System.out.println("F = "+f);
			F += f;
			num ++;
			return f;
		}
		
	}
	
	public int lableTest(int predic,int i){
		if (lable[i] == predic) return 1;
		else return 0;
	}
	public int getLable(int i){
		return lable[i];
	}
	public void printV(int i){
		System.out.println("the "+i+"th : "+v1[i]+" "+v2[i]+" "+v3[i]+" "+v4[i]+" Lable: "+lable[i]);
	}
	public void printF(){
		System.out.println();System.out.println();
		System.out.println("F =  "+F);
		System.out.println("num= "+num);
		System.out.println("average: "+F * 1.0 / num);
		System.out.println();System.out.println();
	}

	public double computeF(double[] ds, int d, int y, int yd) {
		return (ds[4*yd] - ds[4*y] ) * v1[d] + (ds[4*yd + 1] - ds[4*y + 1]) * v2[d]
				+ (ds[4*yd + 2] - ds[4*y + 2]) * v3[d] + (ds[4*yd + 3] - ds[4*y + 3]) * v4[d];
	}
	
	public double computeF(double[] ds, int d, int y) {
		return ds[4 * y] * v1[d] + ds[4 * y + 1] * v2[d] + ds[4 * y + 2]* v3[d] + ds[4 * y + 3] * v4[d];
	}
}