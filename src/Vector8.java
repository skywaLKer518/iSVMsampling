public class Vector8{
	double d[] = new double[8];
	Vector8(){
		setValue(0,0,0,0,0,0,0,0);
	}
	void add(Vector8 a){
		for (int i = 0; i < 8; i ++){
			this.d[i] = this.d[i] + a.d[i];
		}
	}
	void sub(Vector8 a){
		for (int i = 0; i < 8; i ++){
			this.d[i] -= a.d[i];
		}
	}
	void multiply(double a){
		for (int i = 0; i < 8; i ++){
			this.d[i] = this.d[i] * a;
		}
	}
	public double multiply(Vector8 a) {
		double r = 0;
		for (int i = 0; i < 8; i ++){
			r += d[i] * a.d[i];
		}
		return r;
	}
	double multiply(double[] a){
		double r = 0;
		for (int i = 0; i < 8; i++){
			r += a[i] * d[i]; 
		}
		return r;
	}
	double[] getValue(){
		double[] a= new double[8];
		for (int i = 0; i < 8; i ++){
			a[i] = d[i];
		}
		return a;
	}
	double[] getValue(Vector8 v){
		double[] a= new double[8];
		for (int i = 0; i < 8; i ++){
			a[i] = v.d[i];
		}
		return a;
	}
	void getValue(double[] a){
		for (int i = 0; i < 8; i ++){
			a[i] = d[i];
		}
	}
	void setValue(double x0, double x1,double x2, double x3, 
			double x4, double x5,double x6, double x7){
		d[0] = x0;
		d[1] = x1;
		d[2] = x2;
		d[3] = x3;
		d[4] = x4;
		d[5] = x5;
		d[6] = x6;
		d[7] = x7;
		return;
	}
	void setValue(double[] x){
		for (int i = 0; i < 8; i++){
			d[i] = x[i];
		}
		return;
	}
	public void init() {
		setValue(0,0,0,0,0,0,0,0);
	}
	public void print() {
		System.out.print("print vector:\n\t");
		for (int i = 0; i < 8; i ++){
			System.out.print(d[i]+" ");
		}
		System.out.println();
	}
	public void reset(){
		for (int i = 0; i < 8; i++)
			d[i] = 0;
	}
	public int equal(Vector8 v8) {
		for (int i = 0; i < 8; i ++){
			System.out.println(this.d[i]+" "+v8.d[i]);
			if (this.d[i] != v8.d[i]){
				System.out.println("not equal at "+i);
				return -1;
			}
		}
		System.out.println();
		return +1;
	}
	public boolean empty() {
		int a = 0;
		for (int i = 0 ; i < 8; i ++){
			if (d[i] == 0) a++;
			else return false;
		}
		if (a == 8) return true;
		else return false;
	}
}