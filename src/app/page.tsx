'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  DataPoint, 
  SVMResult, 
  KernelType, 
  trainSVM, 
  computeDecisionBoundary,
  formatNumber 
} from '@/lib/svm';
import { 
  Play, 
  RotateCcw, 
  Plus, 
  Minus, 
  Trash2, 
  BookOpen, 
  Calculator,
  BarChart3,
  GitBranch
} from 'lucide-react';

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

const presetData: { [key: string]: DataPoint[] } = {
  simple: [
    { id: '1', x: 1, y: 1, label: 1 },
    { id: '2', x: 2, y: 2, label: 1 },
    { id: '3', x: 1, y: 2, label: -1 },
    { id: '4', x: 2, y: 1, label: -1 },
  ],
  linear: [
    { id: '1', x: 1, y: 3, label: 1 },
    { id: '2', x: 2, y: 4, label: 1 },
    { id: '3', x: 3, y: 5, label: 1 },
    { id: '4', x: 4, y: 2, label: -1 },
    { id: '5', x: 5, y: 1, label: -1 },
    { id: '6', x: 5, y: 3, label: -1 },
  ],
  nonlinear: [
    { id: '1', x: 0, y: 0, label: 1 },
    { id: '2', x: 1, y: 1, label: 1 },
    { id: '3', x: 0.5, y: 0.5, label: 1 },
    { id: '4', x: 2, y: 0, label: -1 },
    { id: '5', x: 2, y: 2, label: -1 },
    { id: '6', x: 1, y: 2, label: -1 },
    { id: '7', x: 0, y: 2, label: -1 },
    { id: '8', x: 2, y: 1, label: -1 },
  ],
};

export default function SVMVisualizationPage() {
  const [points, setPoints] = useState<DataPoint[]>([]);
  const [currentLabel, setCurrentLabel] = useState<1 | -1>(1);
  const [kernelType, setKernelType] = useState<KernelType>('linear');
  const [gamma, setGamma] = useState(1);
  const [C, setC] = useState(100);
  const [result, setResult] = useState<SVMResult | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [showBoundary, setShowBoundary] = useState(true);
  const [activeTab, setActiveTab] = useState('canvas');
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const canvasSize = 500;
  const padding = 40;
  const gridSize = 10;

  const dataRange = {
    minX: 0,
    maxX: 6,
    minY: 0,
    maxY: 6,
  };

  const toCanvasCoord = useCallback((x: number, y: number) => {
    const scaleX = (canvasSize - 2 * padding) / (dataRange.maxX - dataRange.minX);
    const scaleY = (canvasSize - 2 * padding) / (dataRange.maxY - dataRange.minY);
    return {
      cx: padding + (x - dataRange.minX) * scaleX,
      cy: canvasSize - padding - (y - dataRange.minY) * scaleY,
    };
  }, []);

  const toDataCoord = useCallback((cx: number, cy: number) => {
    const scaleX = (canvasSize - 2 * padding) / (dataRange.maxX - dataRange.minX);
    const scaleY = (canvasSize - 2 * padding) / (dataRange.maxY - dataRange.minY);
    return {
      x: (cx - padding) / scaleX + dataRange.minX,
      y: (canvasSize - padding - cy) / scaleY + dataRange.minY,
    };
  }, []);

  // 检查result是否与当前points匹配
  const isResultValid = useCallback(() => {
    if (!result) return false;
    return result.alphas.length === points.length;
  }, [result, points.length]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= gridSize; i++) {
      const x = padding + (i / gridSize) * (canvasSize - 2 * padding);
      const y = padding + (i / gridSize) * (canvasSize - 2 * padding);
      
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, canvasSize - padding);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvasSize - padding, y);
      ctx.stroke();
    }

    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    for (let i = 0; i <= gridSize; i += 2) {
      const val = dataRange.minX + (i / gridSize) * (dataRange.maxX - dataRange.minX);
      const x = padding + (i / gridSize) * (canvasSize - 2 * padding);
      ctx.fillText(val.toFixed(0), x, canvasSize - padding + 20);
    }
    
    ctx.textAlign = 'right';
    for (let i = 0; i <= gridSize; i += 2) {
      const val = dataRange.minY + (i / gridSize) * (dataRange.maxY - dataRange.minY);
      const y = canvasSize - padding - (i / gridSize) * (canvasSize - 2 * padding);
      ctx.fillText(val.toFixed(0), padding - 10, y + 4);
    }

    // 只有当result有效时才绘制决策边界
    if (showBoundary && result && isResultValid() && result.alphas.length > 0 && points.length > 0) {
      const grid = computeDecisionBoundary(points, result.alphas, result.b, kernelType, gamma, 50);
      
      if (grid.length > 0) {
        const scaleX = (canvasSize - 2 * padding) / (dataRange.maxX - dataRange.minX);
        const scaleY = (canvasSize - 2 * padding) / (dataRange.maxY - dataRange.minY);
        
        for (let i = 0; i < grid.length - 1; i++) {
          for (let j = 0; j < grid[i].length - 1; j++) {
            const p1 = grid[i][j];
            
            const cx1 = padding + (p1.x - dataRange.minX) * scaleX;
            const cy1 = canvasSize - padding - (p1.y - dataRange.minY) * scaleY;
            
            const val = p1.value;
            let alpha = Math.min(Math.abs(val) / 2, 0.3);
            
            if (val > 0) {
              ctx.fillStyle = `rgba(34, 197, 94, ${alpha})`;
            } else {
              ctx.fillStyle = `rgba(239, 68, 68, ${alpha})`;
            }
            
            const cellWidth = scaleX * (grid[1][0].x - grid[0][0].x);
            const cellHeight = scaleY * (grid[0][1].y - grid[0][0].y);
            ctx.fillRect(cx1, cy1 - cellHeight, cellWidth, cellHeight);
          }
        }
        
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        for (let i = 0; i < grid.length - 1; i++) {
          for (let j = 0; j < grid[i].length - 1; j++) {
            const cells = [
              grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]
            ];
            
            for (let k = 0; k < 4; k++) {
              const p1 = cells[k];
              const p2 = cells[(k + 1) % 4];
              
              if ((p1.value >= 0 && p2.value < 0) || (p1.value < 0 && p2.value >= 0)) {
                const t = p1.value / (p1.value - p2.value);
                const x = p1.x + t * (p2.x - p1.x);
                const y = p1.y + t * (p2.y - p1.y);
                
                const { cx, cy } = toCanvasCoord(x, y);
                ctx.beginPath();
                ctx.arc(cx, cy, 1, 0, Math.PI * 2);
                ctx.stroke();
              }
            }
          }
        }
        ctx.setLineDash([]);
      }
    }

    points.forEach((point, index) => {
      const { cx, cy } = toCanvasCoord(point.x, point.y);
      
      // 只有当result有效时才检查支持向量
      const isSupportVector = isResultValid() && result?.supportVectors.includes(index);
      
      ctx.beginPath();
      ctx.arc(cx, cy, isSupportVector ? 12 : 8, 0, Math.PI * 2);
      
      if (point.label === 1) {
        ctx.fillStyle = '#22c55e';
      } else {
        ctx.fillStyle = '#ef4444';
      }
      ctx.fill();
      
      if (isSupportVector) {
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(point.label === 1 ? '+' : '-', cx, cy);
    });

    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    
    ctx.beginPath();
    ctx.arc(canvasSize - 100, 20, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#22c55e';
    ctx.fill();
    ctx.fillStyle = '#374151';
    ctx.fillText('正类 (+1)', canvasSize - 88, 24);
    
    ctx.beginPath();
    ctx.arc(canvasSize - 100, 40, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ef4444';
    ctx.fill();
    ctx.fillStyle = '#374151';
    ctx.fillText('负类 (-1)', canvasSize - 88, 44);
    
    ctx.beginPath();
    ctx.arc(canvasSize - 100, 60, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#9ca3af';
    ctx.fill();
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#374151';
    ctx.fillText('支持向量', canvasSize - 88, 64);
  }, [points, result, showBoundary, kernelType, gamma, toCanvasCoord, isResultValid]);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    
    const { x, y } = toDataCoord(cx, cy);
    
    if (x >= dataRange.minX && x <= dataRange.maxX && 
        y >= dataRange.minY && y <= dataRange.maxY) {
      const newPoint: DataPoint = {
        id: generateId(),
        x: Math.round(x * 10) / 10,
        y: Math.round(y * 10) / 10,
        label: currentLabel,
      };
      setPoints(prev => [...prev, newPoint]);
      // 添加新点后清除旧的结果，因为结果不再有效
      setResult(null);
    }
  }, [currentLabel, toDataCoord]);

  const handleTrain = useCallback(() => {
    if (points.length < 2) return;
    
    setIsTraining(true);
    
    setTimeout(() => {
      const svmResult = trainSVM(points, kernelType, gamma, C);
      setResult(svmResult);
      setIsTraining(false);
    }, 100);
  }, [points, kernelType, gamma, C]);

  const handleClear = useCallback(() => {
    setPoints([]);
    setResult(null);
  }, []);

  const handleLoadPreset = useCallback((preset: string) => {
    const data = presetData[preset];
    if (data) {
      setPoints(data.map(p => ({ ...p, id: generateId() })));
      setResult(null);
    }
  }, []);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <GitBranch className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">SVM 可视化教学系统</h1>
                <p className="text-sm text-gray-500">交互式学习支持向量机的原理与计算过程</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-sm">
                {points.length} 个数据点
              </Badge>
              {result && isResultValid() && (
                <Badge variant="secondary" className="text-sm">
                  准确率: {result.accuracy.toFixed(1)}%
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg">交互式画布</CardTitle>
                    <CardDescription>点击画布添加数据点，切换标签类型添加不同类别</CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant={currentLabel === 1 ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setCurrentLabel(1)}
                      className="bg-green-500 hover:bg-green-600"
                    >
                      <Plus className="w-4 h-4 mr-1" /> 正类
                    </Button>
                    <Button
                      variant={currentLabel === -1 ? 'destructive' : 'outline'}
                      size="sm"
                      onClick={() => setCurrentLabel(-1)}
                    >
                      <Minus className="w-4 h-4 mr-1" /> 负类
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div ref={containerRef} className="flex justify-center">
                  <canvas
                    ref={canvasRef}
                    width={canvasSize}
                    height={canvasSize}
                    onClick={handleCanvasClick}
                    className="border border-gray-200 rounded-lg cursor-crosshair shadow-sm"
                  />
                </div>
                
                <div className="flex flex-wrap items-center gap-2 mt-4">
                  <Button onClick={handleTrain} disabled={points.length < 2 || isTraining}>
                    {isTraining ? (
                      <>计算中...</>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-1" /> 训练 SVM
                      </>
                    )}
                  </Button>
                  <Button variant="outline" onClick={handleClear}>
                    <Trash2 className="w-4 h-4 mr-1" /> 清空
                  </Button>
                  <Separator orientation="vertical" className="h-8" />
                  <span className="text-sm text-gray-500">预设数据:</span>
                  <Button variant="outline" size="sm" onClick={() => handleLoadPreset('simple')}>
                    简单示例
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => handleLoadPreset('linear')}>
                    线性可分
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => handleLoadPreset('nonlinear')}>
                    非线性
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Calculator className="w-5 h-5" />
                  计算过程详解
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="canvas">可视化</TabsTrigger>
                    <TabsTrigger value="kernel">核矩阵</TabsTrigger>
                    <TabsTrigger value="alpha">Alpha值</TabsTrigger>
                    <TabsTrigger value="formula">公式推导</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="canvas" className="mt-4">
                    {result && isResultValid() ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="bg-blue-50 rounded-lg p-3">
                            <div className="text-sm text-blue-600 font-medium">迭代次数</div>
                            <div className="text-2xl font-bold text-blue-700">{result.iterations}</div>
                          </div>
                          <div className="bg-green-50 rounded-lg p-3">
                            <div className="text-sm text-green-600 font-medium">支持向量数</div>
                            <div className="text-2xl font-bold text-green-700">{result.supportVectors.length}</div>
                          </div>
                          <div className="bg-purple-50 rounded-lg p-3">
                            <div className="text-sm text-purple-600 font-medium">偏置 b</div>
                            <div className="text-2xl font-bold text-purple-700">{formatNumber(result.b)}</div>
                          </div>
                          <div className="bg-orange-50 rounded-lg p-3">
                            <div className="text-sm text-orange-600 font-medium">准确率</div>
                            <div className="text-2xl font-bold text-orange-700">{result.accuracy.toFixed(1)}%</div>
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="font-medium mb-2">决策函数</h4>
                          <code className="text-sm bg-white px-3 py-2 rounded border block overflow-x-auto">
                            f(x) = Σ αᵢyᵢK(xᵢ, x) + b
                          </code>
                          <p className="text-sm text-gray-600 mt-2">
                            当 f(x) &gt; 0 时预测为正类，f(x) &lt; 0 时预测为负类
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>添加数据点并点击"训练 SVM"查看计算结果</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="kernel" className="mt-4">
                    {result && isResultValid() && result.kernelMatrix.length > 0 ? (
                      <div className="overflow-x-auto">
                        <h4 className="font-medium mb-2">核矩阵 K (Kernel Matrix / Gram Matrix)</h4>
                        <p className="text-sm text-gray-600 mb-3">
                          核矩阵计算公式: K(xᵢ, xⱼ) = {kernelType === 'linear' ? 'xᵢ · xⱼ' : 'exp(-γ||xᵢ - xⱼ||²)'}
                        </p>
                        <table className="min-w-full border-collapse text-sm">
                          <thead>
                            <tr>
                              <th className="border p-2 bg-gray-100"></th>
                              {points.map((_, i) => (
                                <th key={i} className="border p-2 bg-gray-100 font-medium">
                                  x{i + 1}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {result.kernelMatrix.map((row, i) => (
                              <tr key={i}>
                                <td className="border p-2 bg-gray-100 font-medium">x{i + 1}</td>
                                {row.map((val, j) => (
                                  <td 
                                    key={j} 
                                    className={`border p-2 text-center font-mono ${
                                      i === j ? 'bg-blue-50' : ''
                                    }`}
                                  >
                                    {formatNumber(val)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        训练模型后查看核矩阵
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="alpha" className="mt-4">
                    {result && isResultValid() && result.alphas.length > 0 ? (
                      <div className="space-y-4">
                        <h4 className="font-medium">拉格朗日乘子 α (Lagrange Multipliers)</h4>
                        <p className="text-sm text-gray-600">
                          αᵢ &gt; 0 的样本点称为支持向量，它们决定了决策边界的位置
                        </p>
                        <ScrollArea className="h-64 rounded border">
                          <table className="min-w-full">
                            <thead className="sticky top-0 bg-white">
                              <tr>
                                <th className="border-b p-2 text-left bg-gray-50">样本</th>
                                <th className="border-b p-2 text-left bg-gray-50">坐标</th>
                                <th className="border-b p-2 text-left bg-gray-50">标签 y</th>
                                <th className="border-b p-2 text-left bg-gray-50">α</th>
                                <th className="border-b p-2 text-left bg-gray-50">α·y</th>
                                <th className="border-b p-2 text-left bg-gray-50">决策值</th>
                                <th className="border-b p-2 text-left bg-gray-50">类型</th>
                              </tr>
                            </thead>
                            <tbody>
                              {points.map((point, i) => {
                                const isSV = result.supportVectors.includes(i);
                                const alpha = result.alphas[i] ?? 0;
                                const decisionVal = result.decisionValues[i] ?? 0;
                                return (
                                  <tr key={i} className={isSV ? 'bg-blue-50' : ''}>
                                    <td className="border-b p-2">x{i + 1}</td>
                                    <td className="border-b p-2 font-mono">
                                      ({point.x}, {point.y})
                                    </td>
                                    <td className="border-b p-2">
                                      <Badge variant={point.label === 1 ? 'default' : 'destructive'}>
                                        {point.label === 1 ? '+1' : '-1'}
                                      </Badge>
                                    </td>
                                    <td className="border-b p-2 font-mono">
                                      {formatNumber(alpha)}
                                    </td>
                                    <td className="border-b p-2 font-mono">
                                      {formatNumber(alpha * point.label)}
                                    </td>
                                    <td className="border-b p-2 font-mono">
                                      {formatNumber(decisionVal)}
                                    </td>
                                    <td className="border-b p-2">
                                      {isSV ? (
                                        <Badge className="bg-blue-500">支持向量</Badge>
                                      ) : (
                                        <Badge variant="outline">普通点</Badge>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </ScrollArea>
                        
                        <div className="bg-yellow-50 rounded-lg p-3">
                          <p className="text-sm">
                            <strong>约束验证:</strong> Σαᵢyᵢ = {
                              formatNumber(points.reduce((sum, p, i) => 
                                sum + (result.alphas[i] ?? 0) * p.label, 0
                              ))
                            } (应接近 0)
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        训练模型后查看 Alpha 值
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="formula" className="mt-4">
                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium mb-2">1. 原始优化问题</h4>
                        <div className="bg-white p-3 rounded border font-mono text-sm space-y-1">
                          <p>minimize: ½||w||²</p>
                          <p>subject to: yᵢ(w·xᵢ + b) ≥ 1, ∀i</p>
                        </div>
                      </div>
                      
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium mb-2">2. 拉格朗日函数</h4>
                        <div className="bg-white p-3 rounded border font-mono text-sm">
                          <p>L(w, b, α) = ½||w||² - Σαᵢ[yᵢ(w·xᵢ + b) - 1]</p>
                        </div>
                      </div>
                      
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium mb-2">3. 对偶问题</h4>
                        <div className="bg-white p-3 rounded border font-mono text-sm space-y-1">
                          <p>maximize: Σαᵢ - ½ΣᵢΣⱼαᵢαⱼyᵢyⱼK(xᵢ, xⱼ)</p>
                          <p>subject to: Σαᵢyᵢ = 0, αᵢ ≥ 0</p>
                        </div>
                      </div>
                      
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium mb-2">4. KKT 条件</h4>
                        <div className="bg-white p-3 rounded border font-mono text-sm space-y-1">
                          <p>αᵢ ≥ 0 (对偶可行性)</p>
                          <p>yᵢ(w·xᵢ + b) - 1 ≥ 0 (原始可行性)</p>
                          <p>αᵢ[yᵢ(w·xᵢ + b) - 1] = 0 (互补松弛)</p>
                        </div>
                      </div>
                      
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium mb-2">5. 核函数</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div className="bg-white p-3 rounded border">
                            <p className="font-medium text-sm">线性核</p>
                            <p className="font-mono text-sm">K(xᵢ, xⱼ) = xᵢ · xⱼ</p>
                          </div>
                          <div className="bg-white p-3 rounded border">
                            <p className="font-medium text-sm">高斯核 (RBF)</p>
                            <p className="font-mono text-sm">K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">参数设置</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>核函数类型</Label>
                  <RadioGroup 
                    value={kernelType} 
                    onValueChange={(v) => setKernelType(v as KernelType)}
                    className="flex flex-col space-y-1"
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="linear" id="linear" />
                      <Label htmlFor="linear" className="font-normal">线性核 (Linear)</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="rbf" id="rbf" />
                      <Label htmlFor="rbf" className="font-normal">高斯核 (RBF)</Label>
                    </div>
                  </RadioGroup>
                </div>

                <Separator />

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Gamma (γ)</Label>
                    <span className="text-sm text-gray-500">{gamma}</span>
                  </div>
                  <Slider
                    value={[gamma]}
                    onValueChange={([v]) => setGamma(v)}
                    min={0.1}
                    max={10}
                    step={0.1}
                    disabled={kernelType === 'linear'}
                  />
                  <p className="text-xs text-gray-500">
                    高斯核参数，控制决策边界的复杂度
                  </p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>正则化参数 C</Label>
                    <span className="text-sm text-gray-500">{C}</span>
                  </div>
                  <Slider
                    value={[C]}
                    onValueChange={([v]) => setC(v)}
                    min={1}
                    max={1000}
                    step={1}
                  />
                  <p className="text-xs text-gray-500">
                    较大的C值会减少误分类，但可能导致过拟合
                  </p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label>显示选项</Label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="showBoundary"
                      checked={showBoundary}
                      onChange={(e) => setShowBoundary(e.target.checked)}
                      className="rounded border-gray-300"
                    />
                    <Label htmlFor="showBoundary" className="font-normal">
                      显示决策边界
                    </Label>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <BookOpen className="w-5 h-5" />
                  使用说明
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-4 text-sm">
                    <div>
                      <h4 className="font-medium text-gray-900">1. 添加数据点</h4>
                      <p className="text-gray-600 mt-1">
                        选择"正类"或"负类"标签，然后在画布上点击添加数据点。
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900">2. 选择核函数</h4>
                      <p className="text-gray-600 mt-1">
                        线性核适用于线性可分数据，高斯核适用于非线性数据。
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900">3. 调整参数</h4>
                      <p className="text-gray-600 mt-1">
                        Gamma控制高斯核的宽度，C控制对误分类的惩罚程度。
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900">4. 训练模型</h4>
                      <p className="text-gray-600 mt-1">
                        点击"训练SVM"按钮，系统将使用SMO算法求解对偶问题。
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900">5. 分析结果</h4>
                      <p className="text-gray-600 mt-1">
                        查看核矩阵、Alpha值和决策边界，理解SVM的工作原理。
                      </p>
                    </div>
                    
                    <div className="bg-blue-50 rounded p-3">
                      <h4 className="font-medium text-blue-800">关键概念</h4>
                      <ul className="text-blue-700 mt-1 space-y-1 list-disc list-inside">
                        <li>支持向量：αᵢ &gt; 0 的样本点</li>
                        <li>决策边界：f(x) = 0 的等值线</li>
                        <li>间隔：支持向量到决策边界的距离</li>
                      </ul>
                    </div>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <footer className="bg-white border-t border-gray-200 mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4 text-center text-sm text-gray-500">
          SVM 可视化教学系统 | 支持向量机原理演示 | 使用 SMO 算法求解对偶问题
        </div>
      </footer>
    </div>
  );
}
