/**
 * SVM (Support Vector Machine) 计算引擎
 * 支持线性核和高斯核(RBF)
 * 实现SMO算法求解对偶问题
 */

export interface DataPoint {
  id: string;
  x: number;
  y: number;
  label: 1 | -1;
}

export interface SVMResult {
  alphas: number[];
  b: number;
  supportVectors: number[];
  kernelMatrix: number[][];
  decisionValues: number[];
  accuracy: number;
  iterations: number;
}

export type KernelType = 'linear' | 'rbf';

// 线性核函数
function linearKernel(x1: [number, number], x2: [number, number]): number {
  return x1[0] * x2[0] + x1[1] * x2[1];
}

// 高斯核函数 (RBF)
function rbfKernel(x1: [number, number], x2: [number, number], gamma: number): number {
  const dx = x1[0] - x2[0];
  const dy = x1[1] - x2[1];
  return Math.exp(-gamma * (dx * dx + dy * dy));
}

// 计算核矩阵
export function computeKernelMatrix(
  points: DataPoint[],
  kernelType: KernelType,
  gamma: number = 1
): number[][] {
  const n = points.length;
  const K: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const p1: [number, number] = [points[i].x, points[i].y];
      const p2: [number, number] = [points[j].x, points[j].y];
      
      if (kernelType === 'linear') {
        K[i][j] = linearKernel(p1, p2);
      } else {
        K[i][j] = rbfKernel(p1, p2, gamma);
      }
    }
  }
  
  return K;
}

// SMO算法求解SVM对偶问题
export function trainSVM(
  points: DataPoint[],
  kernelType: KernelType,
  gamma: number = 1,
  C: number = 100,
  tolerance: number = 1e-4,
  maxIterations: number = 1000
): SVMResult {
  const n = points.length;
  
  if (n < 2) {
    return {
      alphas: [],
      b: 0,
      supportVectors: [],
      kernelMatrix: [],
      decisionValues: [],
      accuracy: 0,
      iterations: 0
    };
  }
  
  const labels = points.map(p => p.label);
  const hasBothClasses = labels.includes(1) && labels.includes(-1);
  
  if (!hasBothClasses) {
    return {
      alphas: new Array(n).fill(0),
      b: 0,
      supportVectors: [],
      kernelMatrix: computeKernelMatrix(points, kernelType, gamma),
      decisionValues: new Array(n).fill(0),
      accuracy: 100,
      iterations: 0
    };
  }
  
  const K = computeKernelMatrix(points, kernelType, gamma);
  const alphas = new Array(n).fill(0);
  let b = 0;
  
  const H: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      H[i][j] = labels[i] * labels[j] * K[i][j];
    }
  }
  
  const errors = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    errors[i] = decisionFunction(i, alphas, labels, K, b) - labels[i];
  }
  
  let iter = 0;
  let numChanged = 0;
  let examineAll = true;
  
  while ((numChanged > 0 || examineAll) && iter < maxIterations) {
    numChanged = 0;
    
    if (examineAll) {
      for (let i = 0; i < n; i++) {
        numChanged += examineExample(i, n, alphas, labels, H, K, errors, b, C, tolerance);
      }
    } else {
      for (let i = 0; i < n; i++) {
        if (alphas[i] > 0 && alphas[i] < C) {
          numChanged += examineExample(i, n, alphas, labels, H, K, errors, b, C, tolerance);
        }
      }
    }
    
    if (examineAll) {
      examineAll = false;
    } else if (numChanged === 0) {
      examineAll = true;
    }
    
    iter++;
  }
  
  const supportVectors: number[] = [];
  for (let i = 0; i < n; i++) {
    if (alphas[i] > tolerance) {
      supportVectors.push(i);
    }
  }
  
  const decisionValues: number[] = [];
  for (let i = 0; i < n; i++) {
    decisionValues.push(decisionFunction(i, alphas, labels, K, b));
  }
  
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const pred = decisionValues[i] >= 0 ? 1 : -1;
    if (pred === labels[i]) correct++;
  }
  const accuracy = (correct / n) * 100;
  
  return {
    alphas,
    b,
    supportVectors,
    kernelMatrix: K,
    decisionValues,
    accuracy,
    iterations: iter
  };
}

function decisionFunction(
  i: number,
  alphas: number[],
  labels: number[],
  K: number[][],
  b: number
): number {
  let sum = 0;
  for (let j = 0; j < alphas.length; j++) {
    sum += alphas[j] * labels[j] * K[j][i];
  }
  return sum + b;
}

function examineExample(
  i2: number,
  n: number,
  alphas: number[],
  labels: number[],
  H: number[][],
  K: number[][],
  errors: number[],
  b: number,
  C: number,
  tolerance: number
): number {
  const y2 = labels[i2];
  const alpha2 = alphas[i2];
  const E2 = errors[i2];
  const r2 = E2 * y2;
  
  if (!((r2 < -tolerance && alpha2 < C) || (r2 > tolerance && alpha2 > 0))) {
    return 0;
  }
  
  let i1 = -1;
  let maxDiff = 0;
  
  for (let i = 0; i < n; i++) {
    if (alphas[i] > 0 && alphas[i] < C) {
      const E1 = errors[i];
      const diff = Math.abs(E1 - E2);
      if (diff > maxDiff) {
        maxDiff = diff;
        i1 = i;
      }
    }
  }
  
  if (i1 >= 0) {
    if (takeStep(i1, i2, n, alphas, labels, H, K, errors, b, C)) {
      return 1;
    }
  }
  
  const randStart = Math.floor(Math.random() * n);
  for (let i = 0; i < n; i++) {
    const idx = (randStart + i) % n;
    if (alphas[idx] > 0 && alphas[idx] < C && idx !== i2) {
      if (takeStep(idx, i2, n, alphas, labels, H, K, errors, b, C)) {
        return 1;
      }
    }
  }
  
  for (let i = 0; i < n; i++) {
    if (i !== i2) {
      if (takeStep(i, i2, n, alphas, labels, H, K, errors, b, C)) {
        return 1;
      }
    }
  }
  
  return 0;
}

function takeStep(
  i1: number,
  i2: number,
  n: number,
  alphas: number[],
  labels: number[],
  H: number[][],
  K: number[][],
  errors: number[],
  b: number,
  C: number
): boolean {
  if (i1 === i2) return false;
  
  const alpha1 = alphas[i1];
  const alpha2 = alphas[i2];
  const y1 = labels[i1];
  const y2 = labels[i2];
  const E1 = errors[i1];
  const E2 = errors[i2];
  
  let lower: number, upper: number;
  
  if (y1 !== y2) {
    lower = Math.max(0, alpha2 - alpha1);
    upper = Math.min(C, C + alpha2 - alpha1);
  } else {
    lower = Math.max(0, alpha2 + alpha1 - C);
    upper = Math.min(C, alpha2 + alpha1);
  }
  
  if (lower >= upper) return false;
  
  const eta = H[i1][i1] + H[i2][i2] - 2 * H[i1][i2];
  
  let a2: number;
  
  if (eta > 0) {
    a2 = alpha2 + y2 * (E1 - E2) / eta;
    if (a2 < lower) a2 = lower;
    else if (a2 > upper) a2 = upper;
  } else {
    const f1 = y1 * E1 - alpha1 * H[i1][i1] - alpha2 * H[i1][i2];
    const f2 = y2 * E2 - alpha1 * H[i1][i2] - alpha2 * H[i2][i2];
    
    const Lobj = lower * (f1 + f2) + 0.5 * lower * lower * H[i2][i2];
    const Hobj = upper * (f1 + f2) + 0.5 * upper * upper * H[i2][i2];
    
    if (Lobj < Hobj - 1e-10) {
      a2 = lower;
    } else if (Lobj > Hobj + 1e-10) {
      a2 = upper;
    } else {
      a2 = alpha2;
    }
  }
  
  if (Math.abs(a2 - alpha2) < 1e-10) return false;
  
  const a1 = alpha1 + y1 * y2 * (alpha2 - a2);
  
  let bNew: number;
  if (a1 > 0 && a1 < C) {
    bNew = b + E1 + y1 * (a1 - alpha1) * H[i1][i1] + y2 * (a2 - alpha2) * H[i1][i2];
  } else if (a2 > 0 && a2 < C) {
    bNew = b + E2 + y1 * (a1 - alpha1) * H[i1][i2] + y2 * (a2 - alpha2) * H[i2][i2];
  } else {
    bNew = b + (E1 + E2 + y1 * (a1 - alpha1) * H[i1][i1] + y2 * (a2 - alpha2) * H[i2][i2] 
              + y1 * (a1 - alpha1) * H[i1][i2] + y2 * (a2 - alpha2) * H[i2][i2]) / 2;
  }
  
  for (let i = 0; i < n; i++) {
    errors[i] += y1 * (a1 - alpha1) * K[i1][i] + y2 * (a2 - alpha2) * K[i2][i] - (bNew - b);
  }
  
  alphas[i1] = a1;
  alphas[i2] = a2;
  b = bNew;
  
  return true;
}

export function computeDecisionBoundary(
  points: DataPoint[],
  alphas: number[],
  b: number,
  kernelType: KernelType,
  gamma: number,
  resolution: number = 50
): { x: number; y: number; value: number }[][] {
  if (points.length === 0 || alphas.length === 0) {
    return [];
  }
  
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const minX = Math.min(...xs) - 1;
  const maxX = Math.max(...xs) + 1;
  const minY = Math.min(...ys) - 1;
  const maxY = Math.max(...ys) + 1;
  
  const stepX = (maxX - minX) / resolution;
  const stepY = (maxY - minY) / resolution;
  
  const grid: { x: number; y: number; value: number }[][] = [];
  
  for (let i = 0; i <= resolution; i++) {
    const row: { x: number; y: number; value: number }[] = [];
    for (let j = 0; j <= resolution; j++) {
      const x = minX + i * stepX;
      const y = minY + j * stepY;
      
      let value = b;
      for (let k = 0; k < points.length; k++) {
        const p = points[k];
        let kVal: number;
        
        if (kernelType === 'linear') {
          kVal = linearKernel([x, y], [p.x, p.y]);
        } else {
          kVal = rbfKernel([x, y], [p.x, p.y], gamma);
        }
        
        value += alphas[k] * p.label * kVal;
      }
      
      row.push({ x, y, value });
    }
    grid.push(row);
  }
  
  return grid;
}

export function formatNumber(num: number, decimals: number = 4): string {
  return num.toFixed(decimals);
}
