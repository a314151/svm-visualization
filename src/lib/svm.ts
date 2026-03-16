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
  w?: [number, number]; // 线性核的权重向量
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

// SMO状态对象（解决值传递问题）
interface SMOState {
  alphas: number[];
  b: number;
  errors: number[];
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
  
  // 初始化状态对象
  const state: SMOState = {
    alphas: new Array(n).fill(0),
    b: 0,
    errors: new Array(n).fill(0)
  };
  
  // 初始化误差
  for (let i = 0; i < n; i++) {
    state.errors[i] = -labels[i]; // 初始时所有alpha=0，所以f(x)=b=0
  }
  
  let iter = 0;
  let numChanged = 0;
  let examineAll = true;
  
  while ((numChanged > 0 || examineAll) && iter < maxIterations) {
    numChanged = 0;
    
    if (examineAll) {
      for (let i = 0; i < n; i++) {
        numChanged += examineExample(i, n, state, labels, K, C, tolerance);
      }
    } else {
      for (let i = 0; i < n; i++) {
        if (state.alphas[i] > 0 && state.alphas[i] < C) {
          numChanged += examineExample(i, n, state, labels, K, C, tolerance);
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
  
  // 找出支持向量
  const supportVectors: number[] = [];
  for (let i = 0; i < n; i++) {
    if (state.alphas[i] > tolerance) {
      supportVectors.push(i);
    }
  }
  
  // 计算决策值
  const decisionValues: number[] = [];
  for (let i = 0; i < n; i++) {
    decisionValues.push(computeF(i, state.alphas, labels, K, state.b));
  }
  
  // 计算准确率
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const pred = decisionValues[i] >= 0 ? 1 : -1;
    if (pred === labels[i]) correct++;
  }
  const accuracy = (correct / n) * 100;
  
  // 对于线性核，计算权重向量w
  let w: [number, number] | undefined;
  if (kernelType === 'linear') {
    let w1 = 0, w2 = 0;
    for (let i = 0; i < n; i++) {
      w1 += state.alphas[i] * labels[i] * points[i].x;
      w2 += state.alphas[i] * labels[i] * points[i].y;
    }
    w = [w1, w2];
  }
  
  return {
    alphas: state.alphas,
    b: state.b,
    w,
    supportVectors,
    kernelMatrix: K,
    decisionValues,
    accuracy,
    iterations: iter
  };
}

// 计算决策函数值
function computeF(
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

// 检查样本是否违反KKT条件
function examineExample(
  i2: number,
  n: number,
  state: SMOState,
  labels: number[],
  K: number[][],
  C: number,
  tolerance: number
): number {
  const y2 = labels[i2];
  const alpha2 = state.alphas[i2];
  const E2 = state.errors[i2];
  const r2 = E2 * y2;
  
  // 检查KKT条件
  if (!((r2 < -tolerance && alpha2 < C) || (r2 > tolerance && alpha2 > 0))) {
    return 0;
  }
  
  // 选择第二个变量
  let i1 = -1;
  let maxDiff = 0;
  
  // 启发式选择：选择使|E1 - E2|最大的i1
  for (let i = 0; i < n; i++) {
    if (state.alphas[i] > 0 && state.alphas[i] < C) {
      const E1 = state.errors[i];
      const diff = Math.abs(E1 - E2);
      if (diff > maxDiff) {
        maxDiff = diff;
        i1 = i;
      }
    }
  }
  
  if (i1 >= 0 && takeStep(i1, i2, n, state, labels, K, C)) {
    return 1;
  }
  
  // 如果启发式选择失败，遍历所有非边界样本
  const randStart = Math.floor(Math.random() * n);
  for (let i = 0; i < n; i++) {
    const idx = (randStart + i) % n;
    if (state.alphas[idx] > 0 && state.alphas[idx] < C && idx !== i2) {
      if (takeStep(idx, i2, n, state, labels, K, C)) {
        return 1;
      }
    }
  }
  
  // 如果还是失败，遍历所有样本
  for (let i = 0; i < n; i++) {
    if (i !== i2 && takeStep(i, i2, n, state, labels, K, C)) {
      return 1;
    }
  }
  
  return 0;
}

// 执行一步优化
function takeStep(
  i1: number,
  i2: number,
  n: number,
  state: SMOState,
  labels: number[],
  K: number[][],
  C: number
): boolean {
  if (i1 === i2) return false;
  
  const alpha1 = state.alphas[i1];
  const alpha2 = state.alphas[i2];
  const y1 = labels[i1];
  const y2 = labels[i2];
  const E1 = state.errors[i1];
  const E2 = state.errors[i2];
  
  // 计算边界
  let L: number, H: number;
  
  if (y1 !== y2) {
    L = Math.max(0, alpha2 - alpha1);
    H = Math.min(C, C + alpha2 - alpha1);
  } else {
    L = Math.max(0, alpha2 + alpha1 - C);
    H = Math.min(C, alpha2 + alpha1);
  }
  
  if (L >= H) return false;
  
  // 计算eta (二阶导数)
  const k11 = K[i1][i1];
  const k22 = K[i2][i2];
  const k12 = K[i1][i2];
  const eta = k11 + k22 - 2 * k12;
  
  let a2: number;
  
  if (eta > 0) {
    a2 = alpha2 + y2 * (E1 - E2) / eta;
    if (a2 < L) a2 = L;
    else if (a2 > H) a2 = H;
  } else {
    // 计算目标函数在边界点的值
    const f1 = y1 * E1 + state.b - alpha1 * k11 - alpha2 * k12;
    const f2 = y2 * E2 + state.b - alpha1 * k12 - alpha2 * k22;
    
    const Lobj = L * (f2 + y2 * f1) + 0.5 * L * L * k22;
    const Hobj = H * (f2 + y2 * f1) + 0.5 * H * H * k22;
    
    if (Lobj < Hobj - 1e-10) {
      a2 = L;
    } else if (Lobj > Hobj + 1e-10) {
      a2 = H;
    } else {
      a2 = alpha2;
    }
  }
  
  // 检查变化是否足够大
  if (Math.abs(a2 - alpha2) < 1e-10 * (alpha2 + a2 + 1e-10)) {
    return false;
  }
  
  // 计算新的alpha1
  const a1 = alpha1 + y1 * y2 * (alpha2 - a2);
  
  // 更新偏置b
  let bNew: number;
  const b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + state.b;
  const b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + state.b;
  
  if (a1 > 0 && a1 < C) {
    bNew = b1;
  } else if (a2 > 0 && a2 < C) {
    bNew = b2;
  } else {
    bNew = (b1 + b2) / 2;
  }
  
  // 更新误差缓存
  const delta1 = y1 * (a1 - alpha1);
  const delta2 = y2 * (a2 - alpha2);
  
  for (let i = 0; i < n; i++) {
    state.errors[i] += delta1 * K[i1][i] + delta2 * K[i2][i] + (state.b - bNew);
  }
  
  // 更新状态
  state.alphas[i1] = a1;
  state.alphas[i2] = a2;
  state.b = bNew;
  
  return true;
}

// 计算决策边界
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

// 格式化数字
export function formatNumber(num: number, decimals: number = 4): string {
  return num.toFixed(decimals);
}
