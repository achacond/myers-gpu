#include <stdio.h>
#include <sys/file.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define WORD long

#define SIGMA    128
#define BUF_MAX 2048

static int W;

static unsigned WORD All   = -1;
static unsigned WORD Ebit;

static unsigned WORD *TRAN[SIGMA];
static unsigned WORD Pc[SIGMA];
static int seg, rem;

typedef struct 
{
	unsigned WORD P;
	unsigned WORD M;
	int           V;
} Scell;

void search(int ifile,int dif)
{ 
	int num, i, base, diw, a, Cscore;
	Scell *s, *sd;
	unsigned WORD pc, mc;
	register unsigned WORD *e;
	register unsigned WORD P, M, U, X, Y;
	Scell *S, *SE;
	static char buf[BUF_MAX];

	S  = (Scell *) malloc(sizeof(Scell)*seg);
	SE = S + (seg-1);

	diw = dif + W;

	sd = S + (dif-1)/W;
	for (s = S; s <= sd; s++)
	{ 
		s->P = All;
		s->M =  0;
		s->V = ((s-S)+1)*W;
	}

	for (base = 1-rem; (num = read(ifile,buf,BUF_MAX)) > 0; base += num)
	{ 
		i = 0;
		if (sd == S){ 
			P = S->P;
			M = S->M;
			Cscore = S->V;
			for (; i < num; i++){
				a = buf[i];
				U = Pc[a];
				X  = (((U & P) + P) ^ P) | U;
				U |= M;
    
				Y = P;
				P = M | ~ (X | Y);
				M = Y & X;

				if (P & Ebit) Cscore += 1;
              		else if (M & Ebit) Cscore -= 1;

				Y = P << 1;
				P = (M << 1) | ~ (U | Y);
				M = Y & U;

				if (Cscore <= dif) break;
            }

			S->P = P;
			S->M = M;
			S->V = Cscore;

			if (i >= num) continue;
			if (sd == SE) printf("  Match at %d\n",base+i);
            i += 1;
		}

		for (; i < num; i++){ 
			e  = TRAN[buf[i]];
			pc = mc = 0;
			s  = S;
			while (s <= sd){ 
			  U  = *e++;
              P  = s->P;
              M  = s->M;

              Y  = U | mc;
              X  = (((Y & P) + P) ^ P) | Y;
              U |= M;

              Y = P;
              P = M | ~ (X | Y);
              M = Y & X;
    
              Y = (P << 1) | pc;
              s->P = (M << 1) | mc | ~ (U | Y);
              s->M = Y & U;
    
              U = s->V;
              pc = mc = 0;
              if (P & Ebit){ pc = 1; s->V = U+1;}
              else if (M & Ebit){ mc = 1; s->V = U-1; }
              s += 1;
			}

            if (U == dif && (*e & 0x1 | mc) && s <= SE){
				s->P = All;
				s->M = 0;
				if (pc == 1) s->M = 0x1;
				if (mc != 1) s->P <<= 1;
				s->V = U = diw-1;
				sd = s;
            }
            else
            { 
				U = sd->V;
				while (U > diw){
					U = (--sd)->V;
                }
            }
          	if (sd == SE && U <= dif) printf("  Match at %d\n",base+i);
        }

		while (sd > S){ 
				i = sd->V; 
				P = sd->P;
				M = sd->M;
				Y = Ebit;
				for (X = 0; X < W; X++){
					if (P & Y){ 
						i -= 1;
						if (i <= dif) break;
		            }
					else if (M & Y)
		            i += 1;
					Y >>= 1;
				}
				if (i <= dif) break;
				sd -= 1;
		}
    }//for_end

	if (sd == SE)
	{
		P = sd->P;
		M = sd->M;
		U = sd->V;
		for (i = 0; i < rem; i++)
		{
			if (P & Ebit) U -= 1;
          	else if (M & Ebit) U += 1;

			P <<= 1;
			M <<= 1;
			if (U <= dif) printf("  Match at %d\n",base+i);
        }
    }
}
