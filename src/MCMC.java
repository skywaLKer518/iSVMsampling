import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;


public class MCMC {
	private static final int repeatTimes = 5;
	private static final int Infinity = -200;
	private int stateNum = Environment.trainSize;
	private int sampleTimes = Environment.sampleTimes;
	private int[] z = new int[Environment.trainSize];
	private int number[] = new int[stateNum]; // record n_-i,c.  number[MaxCateNumber]
	private int cateNumber = 0; // number of category
	private int cateIndexMax = 0;
	private int observed = 0;
	private int minN = 1;
	private List<Integer> cateAlive = new LinkedList<Integer>(); // store all cate which is associated with at least one data sample
	private double alpha = 1;
	private double[] w = new double[Environment.trainSize * Environment.dataCateNum];
	Vector8[] miu;
	Vector8[] wf;
	private Data data;
	private double[] maxSum = new double[sampleTimes];
	private int rej = 0;
	private int acc = 0;
	
	public MCMC(Data v4, int trainSize, double[] w, double alpha) {
		this.data = v4;
		this.stateNum = trainSize;
		this.alpha = alpha;
		this.w = w;
		//readW();

		miu = new Vector8[Environment.trainSize];
		wf =  new Vector8[Environment.trainSize];
		//
//		init();
		int yy;
		double[] tmp = new double[8];
		for (int i = 0; i < stateNum; i++){
			z[i] = 0;
			miu[i] = new Vector8(); // for each category
			wf[i] = new Vector8(); // for each data
			int yd = ((Vector4) data).getLabel(i);
			if (yd == 0) yy = 1;
			else yy = 0;
			tmp =((Vector4) data).deltaF_d(i);
			wf[i].setValue(tmp);
			wf[i].multiply(w[2 * i + yy]);
			// test when wf = 0;
//			if (wf[i].empty()){
//				System.out.println("ahhhh, this is zero for wf["+i+"]");
//				Vector8 a = new Vector8();
//				a.setValue(((Vector4) data).deltaF_d(i));
//				
//				a.print();
//				Vector8 b = new Vector8();
//				b.add(a);
//				b.multiply(w[2*i+yy]);
//				b.print();
//				System.out.println("w[2*"+i+"+"+yy+"] ="+ w[2*i+yy]);
//			}
		}
		minN = 1;
		// test
//		System.out.println("\nafter init");
//		for (int j = 0; j < 20; j ++){
//			((Vector4) data).printV(j);
//			System.out.print("wf : ");
//			wf[j].print();
//			System.out.println("w[2*"+j+"] = "+w[2*j]);
//			System.out.println("w[2*"+j+"+1] = "+w[2*j+1]);
//		}
		//
		
		
//		Log ww = new Log("oneWsample2.txt");
//		for (int i = 0; i < 200; i++){
//			ww.outln(this.w[i]);
//		}
//		ww.close();
	}
	private void readW() {
		BufferedReader br = null;
		FileInputStream fis = null;
		InputStreamReader isr = null;
		
		try {
			fis = new FileInputStream("oneWsample.txt");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		isr = new InputStreamReader(fis);
		br = new BufferedReader(isr);
		
		try {
			String line = new String();
			for (int i = 0; i < 200; i++){
				line = br.readLine();
				w[i] = Double.parseDouble(line);
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	public void test(){
		System.out.println("this is in test func");
		System.out.println("this is the end in test func");
	}
	
	public void oneSample() {
		int c = 0;
		double old,nnew,accept,r;
		for (int j  = 0; j < stateNum; j ++){
			for (int k = 0; k < repeatTimes; k++){
				c = drawZprior(j);
				if (c == z[j]) continue;
				old = miu[z[j]].multiply(miu[z[j]]) + miu[c].multiply(miu[c]);
				miu[z[j]].sub(wf[j]);
				miu[c].add(wf[j]);
				nnew = miu[z[j]].multiply(miu[z[j]]) + miu[c].multiply(miu[c]);
				accept = 0.5 * (nnew - old);
				if (accept > 0) accept = 1;
				else if (accept < -50) accept = 0;
				else accept = Math.exp(accept);
				r = Math.random();
				if (r <= accept){
					updateState(j,c);
					acc ++;
				}
				else{
					miu[z[j]].add(wf[j]);
					miu[c].sub(wf[j]);
					rej ++;
				}
			}
		}
	}
	public void oneSample2() {
		double [] logP = new double[stateNum],p = new double[stateNum];
		double logPmax = 0, logZ = 0, deltaSigma = 0, sigma = 0, logP_0 = 0, r = 0, F = 0;
		// initial sigma
		for (int t = 1; t <= cateIndexMax; t++){
			sigma += miu[t].multiply(miu[t]);
		}
		sigma *= 0.5;
		for (int i = 0; i < 1; i++){ // TODO
			// sample z[j] for each j
			for (int j = 0; j < stateNum; j++){
				// compute p[k] for each possible k
				int max = 0;
				if (minN > cateIndexMax) max = minN;
				else max = cateIndexMax;
				
				for (int k = 1; k <= max; k++){
					if (k == z[j]){
						logP_0 = Math.log((number[k] - 1) * 1.0 / ((observed * 1.0) + alpha - 1));
						logP[k] = sigma+logP_0;
					}
					else if (number[k] > 0){
						logP_0 = Math.log((number[k]) * 1.0 / ((observed * 1.0) + alpha - 1));
						deltaSigma = (miu[k].minus(miu[z[j]])).multiply(wf[j]) + wf[j].multiply(wf[j]);
						logP[k] = sigma + deltaSigma + logP_0;
					}
					else if (k == minN){
						logP_0 = Math.log( alpha / ((observed * 1.0) + alpha - 1));
						deltaSigma = (miu[k].minus(miu[z[j]])).multiply(wf[j]) + wf[j].multiply(wf[j]);
						logP[k] = sigma + deltaSigma + logP_0;
					}
					else{ // number[k] = 0, k != minN it is a useless component index, which should be deleted
						logP[k] = - Infinity;
					}
				}
				logPmax = -1;
				for (int k = 1; k <= max; k++){
					if (logP[k] > logPmax)
						logPmax = logP[k];
				}
				logZ = logPmax;
				double sumTmp = 0;
				for (int k = 1; k <= max; k++){
					sumTmp += Math.exp(logP[k] - logPmax);
				}
				logZ += Math.log(sumTmp);
				for (int k = 1; k <= max; k++){
					p[k] = Math.exp(logP[k] - logZ);
				}
				
				// standard sampling
				r = Math.random();
				F = 0;
				int choose = 1;
				do{
					F += p[choose];
					if (r < F)
						break;
					choose++;
				}
				while(true);
				
				// update sigma, miu
				if ( choose == z[j] )
					sigma += 0;
				else{
					sigma += (miu[choose].minus(miu[z[j]])).multiply(wf[j]) + wf[j].multiply(wf[j]);
					miu[z[j]].sub(wf[j]);
					miu[choose].add(wf[j]);
				}
				updateState(j,choose);
			}	
		}
		
	}
	
	// using Metropolis Hastings
	public void go(){
		int c = 0;//,yy = 0;;
		double old,nnew;
		double accept = 0, r = 0;
		initialZ();
		Log sample = new Log("res/sampleProcess.txt");
		for (int i = 0; i <sampleTimes; i++ ){
			// debug
			boolean debug = false; // TODO
			double sum = 0;
			if (true){
				if (debug) System.out.println("\nCurrent state is :");
				for (int j = 0; j < stateNum; j ++){
					if (debug) System.out.print(z[j]+" ");
					if (j % 10 == 9)
						if (debug) System.out.println();
				}
				sum = 0;
				for (int t = 1; t <= cateIndexMax; t++){
					sum += miu[t].multiply(miu[t]);
					if (debug) System.out.println("number["+t+"] = : "+number[t]+ "  miu["+t+"]"+"^2: "+miu[t].multiply(miu[t]));
				}
				sum *= 0.5;
				if (debug) System.out.println(i+"th iteration, sum of miu = : "+sum);
				maxSum[i] = sum;
			}
			sample.outln("\nCurrent state is :");
			for (int j = 0; j < stateNum; j ++){
				sample.out(z[j]+" ");
				if (j % 10 == 9)
					sample.newline();
			}
			for (int t = 1; t <= cateIndexMax; t++){
				sample.outln("number["+t+"] = : "+number[t]+ "  miu["+t+"]"+"^2: "+miu[t].multiply(miu[t]));
			}
			sample.outln(i+"th iteration, sum of miu = : "+sum);
			// ~debug
			for (int j = 0; j < stateNum; j ++){
				for (int k = 0; k < repeatTimes; k++){
					// sampleCandidate(j);
					c = drawZprior(j);
					if (c == z[j]) continue;
					// compute a*
					old = miu[z[j]].multiply(miu[z[j]]) + miu[c].multiply(miu[c]);
					miu[z[j]].sub(wf[j]);
					miu[c].add(wf[j]);
					nnew = miu[z[j]].multiply(miu[z[j]]) + miu[c].multiply(miu[c]);
					accept = 0.5 * (nnew - old);
					
					// debug
					boolean accDebug = false;
					if (accDebug){
						System.out.println(j +"th z:\n from "+z[j]+ " to "+ c +"   (old: "+old +" ; new: "+nnew );
						System.out.println("accept =  : "+accept);
					}
					sample.outln(j +"th z:\n from "+z[j]+ " to "+ c +"   (old: "+old +" ; new: "+nnew );
					sample.outln("accept =  : "+accept);
					// ~debug
					// do
					if (accept > 0) accept = 1;
					else if (accept < -50) accept = 0;
					else accept = Math.exp(accept);
					
					r = Math.random();
					if (r <= accept){
						updateState(j,c);
						acc ++;
						sample.outln("ACC");
					}
					else{
						miu[z[j]].add(wf[j]);
						miu[c].sub(wf[j]);
						rej ++;
						sample.outln("REJ");
					}
				}
				if (observed < stateNum -1)
					observed ++;
			}
		}
		Log s = new Log("res/probLog.txt");
		Log v = new Log("res/prob.txt");
		s.outln("accept number: "+acc);
		s.outln("reject number: "+rej);
		for (int i = 0;i < sampleTimes; i++){
//			s.outln("sum : "+maxSum[i]);
			s.outln(maxSum[i]);
			v.outln(Math.exp(maxSum[i]));
		}
		s.close();
		v.close();
		sample.close();
	}
	
	// using Gibbs Sampling
	public void go2(){
		double [] logP = new double[stateNum],p = new double[stateNum];
		double logPmax = 0, logZ = 0, deltaSigma = 0, sigma = 0, logP_0 = 0, r = 0, F = 0;
		
		initialZ();
		Log sample = new Log("res/sampleProcess.txt");
		// initial sigma
		for (int t = 1; t <= cateIndexMax; t++){
			sigma += miu[t].multiply(miu[t]);
		}
		sigma *= 0.5;
		
		Log condition = new Log("conditionalP.txt");
		Log certainZ = new Log("res/certainZ.txt");
		for (int i = 0; i < sampleTimes; i++){
			
			// debug
			boolean debug = true; // TODO
			double sum = 0;
			if (debug) System.out.println("\nCurrent state is :");
			sample.outln("\nCurrent state is :");
			for (int j = 0; j < stateNum; j ++){
				if (debug) System.out.print(z[j]+" ");
				sample.out(z[j]+" ");
				if (j % 10 == 9){
					if (debug) System.out.println();
					sample.newline();
				}
			}
			sum = 0;
			for (int t = 1; t <= cateIndexMax; t++){
				sum += miu[t].multiply(miu[t]);
				if (debug) System.out.println("number["+t+"] = : "+number[t]+ "  miu["+t+"]"+"^2: "+miu[t].multiply(miu[t]));
				sample.outln("number["+t+"] = : "+number[t]+ "  miu["+t+"]"+"^2: "+miu[t].multiply(miu[t]));
			}
			sum *= 0.5;
			if (debug) System.out.println(i+"th iteration, sum of miu = : "+sum);
			sample.outln(i+"th iteration, sum of miu = : "+sum);
			if (sum - sigma > 1){
				System.out.println("sum = "+sum +"  sigma = "+sigma);
				System.exit(-1);
			}
			maxSum[i] = sum;
			// ~debug
						
			// sample z[j] for each j
			for (int j = 0; j < stateNum; j++){
				if ( j == 5 )certainZ.outln(z[j]);
				// compute p[k] for each possible k
				/*
				 * a, logp
				 * b, p
				 */
				int max = 0;
				if (minN > cateIndexMax) max = minN;
				else max = cateIndexMax;
				
				for (int k = 1; k <= max; k++){
					if (k == z[j]){
						logP_0 = Math.log((number[k] - 1) * 1.0 / ((observed * 1.0) + alpha - 1));
						logP[k] = sigma+logP_0;
					}
					else if (number[k] > 0){
						logP_0 = Math.log((number[k]) * 1.0 / ((observed * 1.0) + alpha - 1));
						deltaSigma = (miu[k].minus(miu[z[j]])).multiply(wf[j]) + wf[j].multiply(wf[j]);
						logP[k] = sigma + deltaSigma + logP_0;
					}
					else if (k == minN){
						logP_0 = Math.log( alpha / ((observed * 1.0) + alpha - 1));
						deltaSigma = (miu[k].minus(miu[z[j]])).multiply(wf[j]) + wf[j].multiply(wf[j]);
						logP[k] = sigma + deltaSigma + logP_0;
					}
					else{ // number[k] = 0, k != minN it is a useless component index, which should be deleted
						logP[k] = - Infinity;
					}
				}
				logPmax = -1;
				for (int k = 1; k <= max; k++){
					if (logP[k] > logPmax)
						logPmax = logP[k];
				}
				logZ = logPmax;
				double sumTmp = 0;
				for (int k = 1; k <= max; k++){
					sumTmp += Math.exp(logP[k] - logPmax);
				}
				logZ += Math.log(sumTmp);
				for (int k = 1; k <= max; k++){
					p[k] = Math.exp(logP[k] - logZ);
				}
				
				// test
				condition.outln("for "+j);
				for (int k = 1; k <= max; k++){
					condition.outln(p[k]);
				} // ~test
				
				// standard sampling
				r = Math.random();
				F = 0;
				int choose = 1;
				do{
					F += p[choose];
					if (r < F)
						break;
					choose++;
				}
				while(true);
				
				// update sigma, miu
				if ( choose == z[j] )
					sigma += 0;
				else{
					sigma += (miu[choose].minus(miu[z[j]])).multiply(wf[j]) + wf[j].multiply(wf[j]);
					miu[z[j]].sub(wf[j]);
					miu[choose].add(wf[j]);
				}
				updateState(j,choose);
			}	
		}
		
		Log s = new Log("res/probLog.txt");
		Log v = new Log("res/prob.txt");
		s.outln("accept number: "+acc);
		s.outln("reject number: "+rej);
		for (int i = 0;i < sampleTimes; i++){
//			s.outln("sum : "+maxSum[i]);
			s.outln(maxSum[i]);
			v.outln(Math.exp(maxSum[i]));
		}
		s.close();
		v.close();
		
		sample.close();
		condition.close();
		certainZ.close();
	}
	


	// correct: use c to update z[i] with c
	private void updateState(int i, int c){
		int previous = z[i];
		// case 0: state[i] == 0;
		//          change state[i] to c; if (number[c]>0), else cateNumber++ ;number[c]++; if (c > cateIndexMax) cateIndexMax = c; MinN++ until MinN not used;
		//          if cateAlive contains c , else cateAlive add c;
		if (previous == 0){				// state
			z[i] = c;
			if (number[c] <= 0){		// cateNumber,number[]
				cateNumber++;
				number[c]++;
			}
			else{
				number[c]++;
			}
			if (c > cateIndexMax)			// cateIndexMax
				cateIndexMax = c;
			if (number[minN]!=0){
				do{ // minN
					minN ++;
				}while (number[minN]!=0);
			}
			if (!cateAlive.contains(c)){  // cateAlive
				cateAlive.add(c);
			}
		}
		else {
			// case 1: state i exists, so does c
			//		    change all state with the same state[i] to be c;number[c] += number[i],number[i] = 0;cateNumber--;if (state[i] == cateIndexMax) 
			//          cateIndexMax -- until number[cateIndexMax - 1]>0; MinN: i < MinN then MinN = i;  if cateAlive contains state[i], remove it.
			// case 2: state i exists, but not c  : new category
			//			change all state with state[i] to c; number[c] = number[i],number[i] = 0;if (state[i] == cateIndexMax)cateIndexMax -- until number[cateIndexMax - 1]>0;
			//          MinN: if i < MinN then MinN = i; if cateAlive contains state[i], remove it. cateAlive adds c;
			
			if (number[c] > 0) {// case 1
				if (number[previous]==1){
					cateNumber--;
					if (previous < minN) minN = previous;
					if (cateIndexMax == previous){
						do {
							cateIndexMax--;
						}while(number[cateIndexMax] == 0);
					}
				}
			}
			else{
				cateNumber++;
				if (number[previous]==1){
					cateNumber--;
					if (previous < c){
						minN = previous;
					}
					else{// previous > c. at this time c = minN for sure.
						do { // minN
							minN ++;
						}while( number[minN]!=0 && minN != previous);
					}	
				}
			}
			z[i] = c;
			number[c] ++;
			number[previous] --;
			while (number[minN]!=0){
				minN++;
			}
			if (c > cateIndexMax){  // TODO
				cateIndexMax = c;
			}
			if (number[previous] == 0 && cateAlive.contains(previous)){
				int index = cateAlive.indexOf(previous);
				cateAlive.remove(index);
			}
			if (!cateAlive.contains(c)){
				cateAlive.add(c);
			}
			
		}
	}

	private int drawZprior(int j){
		int cur = z[j];
		int n;
		double p = 0,F = 0;
		double r = Math.random();
		Iterator<Integer> it = cateAlive.iterator();
		while (it.hasNext()){
			n = (Integer)it.next();             // the nth category
//			System.out.println("n = "+n);
			if (n == cur)
				p = (number[n] - 1) * 1.0 / ((observed * 1.0) + alpha - 1);
			else
				p = number[n] * 1.0 / ((observed * 1.0) + alpha - 1);
//			System.out.println("F and p = "+F+" , "+p);
			F += p;
			if (r < F)
				return n;
		}
		// a new category
		return minN;
	}
	
	void init(){
		int yy;
		double[] tmp = new double[8];
		for (int i = 0; i < stateNum; i++){
			z[i] = 0;
			miu[i] = new Vector8(); // for each category
			wf[i] = new Vector8(); // for each data
			int yd = ((Vector4) data).getLabel(i);
			if (yd == 0) yy = 1;
			else yy = 0;
			tmp =((Vector4) data).deltaF_d(i);
			wf[i].setValue(tmp);
			wf[i].multiply(w[2 * i + yy]);
		}
		minN = 1;
	}

	private void initialZ() {
		int m = 0;
		// z[0-49] = 1 z[50-99] = 2 minN = 3
		// miu[1] miu[2]
		/*
		for (int j = 0; j < stateNum; j++){
			m = j / 50 + 1;
			z[j] = m;
			number[m]++;
		}
		*/
		for (int j = 0; j < stateNum; j++){
			number[j] = 0;
		}
		for (int j = 0; j < stateNum; j++){
			m = j / 20 + 1;
			z[j] = m;
			number[m]++;
		}
		minN = 6;
		cateAlive.clear();
		cateAlive.add(1);cateAlive.add(2);
		cateAlive.add(3);cateAlive.add(4);cateAlive.add(5);
		cateIndexMax = 5;
		cateNumber = 5;
		observed = 100;
		
		shuffleZ();
//		double[] tmp = new double[8];
//		int yy = 0;
//		Vector8 r = new Vector8();
		for (int j = 0; j < stateNum; j++){
			miu[z[j]].add(wf[j]);
		}
	}
	
	
	private void shuffleZ() {
		int previous = 0;
		for (int i = 0; i < stateNum * 100; i++){
			for (int j = 0; j < stateNum; j++){
				previous = z[j];
				double r = Math.random();
				if (r < 0.2)
					z[j] = 1;
				else if (r < 0.4)
					z[j] = 2;
				else if (r < 0.6)
					z[j] = 3;
				else if (r < 0.8)
					z[j] = 4;
				else z[j] = 5;
				
				if (z[j] != previous){
					number[previous]--;
					number[z[j]]++;
				}	
			}
		}
	}
	public int[] getZ() {
		int [] a = new int[stateNum];
		for (int i = 0; i < stateNum; i++){
			a[i] = z[i];
		}
		return a;
	}
	public int getCateNumer() {
		return cateIndexMax;
	}
	public Vector8 getMiu(int l) {
		return miu[l];
	}
	public int[] getNumberEachCate() {
		int[] r = new int[cateIndexMax+1];
		for (int i = 1; i <= cateIndexMax; i++){
			r[i] = number[i];
		}
		return r;
	}

}
