Для изображений звездного неба требуется найти наиболее яркую прямоугольную область. 

Последовательность преобразований, которые требуется произвести с изображением:
1) загрузить в память для дальнейшей работы
2) преобразовать изображение из цветного в градации серого
3) полученную матрицу, назовем ее M, нужно "нормализовать". Т.е. требуется посчитать максимальное значение всех элементов матрицы М, будем обозначать его как max(M). Нормализованная матрица N будет вычисляться как: N[i][j] = M[i][j] - (max(M) / 2) 
или какая-то похожая метрика.
4) теперь с помощью алгоритма поиска подматрицы с максимальной суммой, мы можем определить наиболее яркую область исходной матрицы. Нужно провести вычисление подматрицы над матрицей N используя 
** либо предпосчитанные 2d prefix sums: https://www.youtube.com/watch?v=WibxoqMSMCw или https://en.wikipedia.org/wiki/Summed-area_table
** либо 2d алгоритм Кадане: https://www.youtube.com/watch?v=-FgseNO-6Gk или http://e-maxx.ru/algo/maximum_average_segment
5) опционально. если размеры изображения достаточно большие, то алгоритм поиска подматрицы будет работать долго. Можно уменьшить разрешение исходного изображания, например в 4 раза.
6) Если обозначать полученную подматрицу с максимальной суммой как {[x1,y1], [x2,y2]}, где x1,y1 и x2, y2 - координаты левого верхнего и правого нижнего элементов подматрицы, то на исходном изображении нужно отобразить прямоугольник с таким координатами. Или можно нарисовать окружность с центром в точке [x2-x1/2;y2-y1/2] и радиусом max(x2-x1, y2-y1)/2


В подпапке ./res расположены реальные изображение телескопа, с которыми можно работать