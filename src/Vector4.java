/*
 * setting 1
 * u1 ~ N(1,1),u2 ~ N(2,1)
 * log(sigma^2) ~ N(0,2^2)
 * 
 */
class Vector4 extends Data{
	private double v1[],v2[],v3[],v4[];
	private int lable[];
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
		System.out.println("Date regeneration completed!");
	}

	private int Lable(int i) {
		double a,b,c,theta,p;
		a = sampleNormalUnivariate(1,0.5*0.5);
		b = sampleNormalUnivariate(1,0.5*0.5);
		c = sampleNormalUnivariate(1,0.5*0.5);
		theta = - a * Math.sin(v1[i] * v1[i] * v1[i] + 1.2) -v1[i] * Math.cos(b * v2[i] + 0.7) - c * v3[i] + 2;
		p = 1 / (1 + Math.exp(-theta));
		if (Math.random() < p){
			return 1;
		}
		else
			return 0;
	}
	
	public double disFunc(double[] eta,int i){
		if (lable[i] == 0){// then f(y,x) = (x[1],x[2],x[3],x[4],0,0,0,0)
			return (eta[0] * v1[i] + eta[1] * v2[i] + eta[2] * v3[i] + eta[3] * v4[i]);
		}
		else{ // then f(y,x) = (0,0,0,0,x[1],x[2],x[3],x[4])
			return (eta[4] * v1[i] + eta[5] * v2[i] + eta[6] * v3[i] + eta[7] * v4[i]);
		}
	}
	
	public double disFunc(double[] eta,int i,int y){
		if (y == 0){// then f(y,x) = (x[1],x[2],x[3],x[4],0,0,0,0)
			return (eta[0] * v1[i] + eta[1] * v2[i] + eta[2] * v3[i] + eta[3] * v4[i]);
		}
		else{ // then f(y,x) = (0,0,0,0,x[1],x[2],x[3],x[4])
			return (eta[4] * v1[i] + eta[5] * v2[i] + eta[6] * v3[i] + eta[7] * v4[i]);
		}
	}
	
	public int lableTest(int predic,int i){
		if (lable[i] == predic) return 1;
		else return 0;
	}
	public void printV(int i){
		System.out.println("the "+i+"th : "+v1[i]+" "+v2[i]+" "+v3[i]+" "+v4[i]+" Lable: "+lable[i]);
	}
	
}