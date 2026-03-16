/**
 * SVM (Support Vector Machine) 计算引擎
 * 支持线性核和高斯核(RBF)
 * 实现简化版SMO算法求解对偶问题
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
  w?: [number, number];
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

// 简化版SMO算法 - 更可靠的实现
export function trainSVM(
  points: DataPoint[],
  kernelType: KernelType,
  gamma: number = 1,
  C: number = 100,
  tolerance: number = 1e-3,
  maxIterations: number = 5000
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
  
  // 初始化
  const alphas = new Array(n).fill(0);
  let b = 0;
  
  // 计算决策函数值
  const computeF = (i: number): number => {
    let sum = b;
    for (let j = 0; j < n; j++) {
      sum += alphas[j] * labels[j] * K[j][i];
    }
    return sum;
  };
  
  // 计算误差
  const computeE = (i: number): number => {
    return computeF(i) - labels[i];
  };
  
  // 简化版SMO主循环
  let iter = 0;
  let changed = true;
  
  while (changed && iter < maxIterations) {
    changed = false;
    
    for (let i = 0; i < n; i++) {
      const Ei = computeE(i);
      const yi = labels[i];
      
      // 检查KKT条件
      const kktViolation = (yi * Ei < -tolerance && alphas[i] < C) || 
                           (yi * Ei > tolerance && alphas[i] > 0);
      
      if (!kktViolation) continue;
      
      // 选择第二个alpha
      let j = i;
      while (j === i) {
        j = Math.floor(Math.random() * n);
      }
      
      const Ej = computeE(j);
      const yj = labels[j];
      
      // 保存旧的alpha值
      const alpha_i_old = alphas[i];
      const alpha_j_old = alphas[j];
      
      // 计算边界
      let L: number, H: number;
      if (yi !== yj) {
        L = Math.max(0, alpha_j_old - alpha_i_old);
        H = Math.min(C, C + alpha_j_old - alpha_i_old);
      } else {
        L = Math.max(0, alpha_j_old + alpha_i_old - C);
        H = Math.min(C, alpha_j_old + alpha_i_old);
      }
      
      if (L >= H) continue;
      
      // 计算eta
      const eta = 2 * K[i][j] - K[i][i] - K[j][j];
      
      if (eta >= 0) continue;
      
      // 更新alpha_j
      alphas[j] = alpha_j_old - yj * (Ei - Ej) / eta;
      
      // 裁剪
      if (alphas[j] > H) alphas[j] = H;
      else if (alphas[j] < L) alphas[j] = L;
      
      // 检查变化是否足够大
      if (Math.abs(alphas[j] - alpha_j_old) < 1e-5) continue;
      
      // 更新alpha_i
      alphas[i] = alpha_i_old + yi * yj * (alpha_j_old - alphas[j]);
      
      // 更新b
      const b1 = b - Ei - yi * (alphas[i] - alpha_i_old) * K[i][i] 
                       - yj * (alphas[j] - alpha_j_old) * K[i][j];
      const b2 = b - Ej - yi * (alphas[i] - alpha_i_old) * K[i][j] 
                       - yj * (alphas[j] - alpha_j_old) * K[j][j];
      
      if (alphas[i] > 0 && alphas[i] < C) {
        b = b1;
      } else if (alphas[j] > 0 && alphas[j] < C) {
        b = b2;
      } else {
        b = (b1 + b2) / 2;
      }
      
      changed = true;
    }
    
    iter++;
  }
  
  // 找出支持向量
  const supportVectors: number[] = [];
  for (let i = 0; i < n; i++) {
    if (alphas[i] > tolerance) {
      supportVectors.push(i);
    }
  }
  
  // 计算决策值
  const decisionValues: number[] = [];
  for (let i = 0; i < n; i++) {
    decisionValues.push(computeF(i));
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
      w1 += alphas[i] * labels[i] * points[i].x;
      w2 += alphas[i] * labels[i] * points[i].y;
    }
    w = [w1, w2];
  }
  
  return {
    alphas,
    b,
    w,
    supportVectors,
    kernelMatrix: K,
    decisionValues,
    accuracy,
    iterations: iter
  };
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
