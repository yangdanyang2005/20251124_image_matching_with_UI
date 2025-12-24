#include "_image_matching_with_UI.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <filesystem>

// ==========================================
// 核心算法静态工具函数
// ==========================================

// 1. 双线性插值与边界检查
static inline bool insideForBilinear(int cols, int rows, double x, double y) {
    // 留出 1 像素边缘以进行插值
    return (x >= 0.0 && y >= 0.0 && x <= (double)cols - 2.0 && y <= (double)rows - 2.0);
}

static inline double bilinearU8(const Mat& imgU8, double x, double y) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    double dx = x - x0;
    double dy = y - y0;

    double v00 = (double)imgU8.at<uchar>(y0, x0);
    double v10 = (double)imgU8.at<uchar>(y0, x0 + 1);
    double v01 = (double)imgU8.at<uchar>(y0 + 1, x0);
    double v11 = (double)imgU8.at<uchar>(y0 + 1, x0 + 1);

    return (1.0 - dx) * (1.0 - dy) * v00 +
        (dx) * (1.0 - dy) * v10 +
        (1.0 - dx) * (dy)*v01 +
        (dx) * (dy)*v11;
}

static inline double bilinearF64(const Mat& imgF64, double x, double y) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    double dx = x - x0;
    double dy = y - y0;

    double v00 = imgF64.at<double>(y0, x0);
    double v10 = imgF64.at<double>(y0, x0 + 1);
    double v01 = imgF64.at<double>(y0 + 1, x0);
    double v11 = imgF64.at<double>(y0 + 1, x0 + 1);

    return (1.0 - dx) * (1.0 - dy) * v00 +
        (dx) * (1.0 - dy) * v10 +
        (1.0 - dx) * (dy)*v01 +
        (dx) * (dy)*v11;
}

static inline bool patchInside(const Mat& img, int x, int y, int halfWin) {
    return (x - halfWin >= 0 && y - halfWin >= 0 &&
        x + halfWin < img.cols && y + halfWin < img.rows);
}

// 2. NCC 计算核心
static inline bool computeNCC(const Mat& L, const Mat& R, int xl, int yl, int xr, int yr,
    int halfWin, double& outRho) {
    if (!patchInside(L, xl, yl, halfWin) || !patchInside(R, xr, yr, halfWin)) return false;

    const int w = 2 * halfWin + 1;
    const int N = w * w;

    double sumL = 0.0, sumR = 0.0;
    double sumLL = 0.0, sumRR = 0.0, sumLR = 0.0;

    for (int dy = -halfWin; dy <= halfWin; ++dy) {
        const uchar* pL = L.ptr<uchar>(yl + dy);
        const uchar* pR = R.ptr<uchar>(yr + dy);
        for (int dx = -halfWin; dx <= halfWin; ++dx) {
            double vL = (double)pL[xl + dx];
            double vR = (double)pR[xr + dx];
            sumL += vL; sumR += vR;
            sumLL += vL * vL;
            sumRR += vR * vR;
            sumLR += vL * vR;
        }
    }

    double meanL = sumL / (double)N;
    double meanR = sumR / (double)N;

    double varL = (sumLL / (double)N) - meanL * meanL;
    double varR = (sumRR / (double)N) - meanR * meanR;
    if (varL <= 1e-12 || varR <= 1e-12) return false;

    double cov = (sumLR / (double)N) - meanL * meanR;
    outRho = cov / std::sqrt(varL * varR);
    return true;
}

// 3. NCC 搜索结构与函数
struct NCCBest2 {
    double bestRho = -1e9;
    double secondRho = -1e9;
    Point bestPt{ 0, 0 };
    bool ok = false;
};

static NCCBest2 searchBest2NCC(const Mat& L, const Mat& R,
    int xl, int yl,
    int xr_est, int yr_est,
    int halfMatch, int halfSearch) {
    NCCBest2 res;
    for (int dy = -halfSearch; dy <= halfSearch; ++dy) {
        for (int dx = -halfSearch; dx <= halfSearch; ++dx) {
            int xr = xr_est + dx;
            int yr = yr_est + dy;
            double rho = -1e9;
            if (!computeNCC(L, R, xl, yl, xr, yr, halfMatch, rho)) continue;
            if (rho > res.bestRho) {
                res.secondRho = res.bestRho;
                res.bestRho = rho;
                res.bestPt = Point(xr, yr);
                res.ok = true;
            }
            else if (rho > res.secondRho) {
                res.secondRho = rho;
            }
        }
    }
    return res;
}

// 4. LSM 求解器
bool solveDampedLeastSquares(const Mat& C, const Mat& L, Mat& dx, double lambda) {
    Mat CtC = C.t() * C;
    Mat CtL = C.t() * L;
    Mat A = CtC.clone();
    for (int i = 0; i < A.rows; ++i) {
        A.at<double>(i, i) *= (1.0 + lambda);
    }
    return cv::solve(A, CtL, dx, DECOMP_SVD);
}

static double getPatchMean(const vector<double>& vec) {
    if (vec.empty()) return 0.0;
    double sum = 0.0;
    for (double v : vec) sum += v;
    return sum / (double)vec.size();
}

// ==========================================
// Qt 类实现
// ==========================================

_image_matching_with_UI::_image_matching_with_UI(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    imageScene = new QGraphicsScene(this);
    imageView = nullptr;

    setupCustomUI();
    applyHammerParams();
}

_image_matching_with_UI::~_image_matching_with_UI() {
    if (imageScene) {
        imageScene->clear();
        delete imageScene;
    }
}

void _image_matching_with_UI::setupCustomUI() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    QSplitter* splitter = new QSplitter(Qt::Horizontal);

    // --- 左侧：控制面板 ---
    QWidget* controlPanel = new QWidget();
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);

    // 1. 文件选择
    QGroupBox* grpFiles = new QGroupBox("文件路径");
    QGridLayout* layoutFiles = new QGridLayout(grpFiles);
    pathLeftEdit = new QLineEdit();
    pathRightEdit = new QLineEdit();
    pathLeftEdit->setMinimumWidth(250);
    pathRightEdit->setMinimumWidth(250);
    QPushButton* btnLeft = new QPushButton("...");
    QPushButton* btnRight = new QPushButton("...");
    btnLeft->setFixedWidth(40);
    btnRight->setFixedWidth(40);
    connect(btnLeft, &QPushButton::clicked, this, &_image_matching_with_UI::selectLeftImage);
    connect(btnRight, &QPushButton::clicked, this, &_image_matching_with_UI::selectRightImage);
    layoutFiles->addWidget(new QLabel("左影像:"), 0, 0);
    layoutFiles->addWidget(pathLeftEdit, 0, 1);
    layoutFiles->addWidget(btnLeft, 0, 2);
    layoutFiles->addWidget(new QLabel("右影像:"), 1, 0);
    layoutFiles->addWidget(pathRightEdit, 1, 1);
    layoutFiles->addWidget(btnRight, 1, 2);
    layoutFiles->setColumnStretch(1, 1);
    controlLayout->addWidget(grpFiles);

    // 2. 参数设置
    QGroupBox* grpParams = new QGroupBox("算法参数");
    QGridLayout* pLayout = new QGridLayout(grpParams);

    spinMoWin = createSpinBox(3, 21, 7, "特征提取窗口");
    spinMoT = new QDoubleSpinBox(); spinMoT->setRange(100, 100000); spinMoT->setValue(1200);
    spinMoNms = createSpinBox(10, 200, 50, "NMS 范围");

    spinOffX = createSpinBox(-5000, 5000, 0, "X 偏移 (Offset X)");
    spinOffY = createSpinBox(-5000, 5000, 0, "Y 偏移 (Offset Y)");
    spinSearch = createSpinBox(5, 100, 25, "搜索窗口 (Search)");
    spinMatch = createSpinBox(3, 50, 9, "匹配窗口 (NCC)");
    spinLsmWin = createSpinBox(5, 50, 25, "LSM 窗口");

    int row = 0;
    pLayout->addWidget(new QLabel("<b>[特征提取 Shi-Tomasi]</b>"), row++, 0, 1, 2);
    pLayout->addWidget(new QLabel("窗口大小:"), row, 0); pLayout->addWidget(spinMoWin, row++, 1);
    pLayout->addWidget(new QLabel("质量阈值 (T):"), row, 0); pLayout->addWidget(spinMoT, row++, 1);
    pLayout->addWidget(new QLabel("最小间距 (NMS):"), row, 0); pLayout->addWidget(spinMoNms, row++, 1);

    pLayout->addWidget(new QLabel("<b>[NCC 粗匹配]</b>"), row++, 0, 1, 2);
    pLayout->addWidget(new QLabel("预估偏移 X:"), row, 0); pLayout->addWidget(spinOffX, row++, 1);
    pLayout->addWidget(new QLabel("预估偏移 Y:"), row, 0); pLayout->addWidget(spinOffY, row++, 1);
    pLayout->addWidget(new QLabel("搜索范围:"), row, 0); pLayout->addWidget(spinSearch, row++, 1);
    pLayout->addWidget(new QLabel("匹配窗口:"), row, 0); pLayout->addWidget(spinMatch, row++, 1);

    pLayout->addWidget(new QLabel("<b>[LSM 精匹配]</b>"), row++, 0, 1, 2);
    pLayout->addWidget(new QLabel("窗口大小:"), row, 0); pLayout->addWidget(spinLsmWin, row++, 1);
    controlLayout->addWidget(grpParams);

    // 3. 预设按钮
    QHBoxLayout* presetLayout = new QHBoxLayout();
    QPushButton* btnHammer = new QPushButton("预设: Hammer");
    QPushButton* btnTown = new QPushButton("预设: Town");
    connect(btnHammer, &QPushButton::clicked, this, &_image_matching_with_UI::applyHammerParams);
    connect(btnTown, &QPushButton::clicked, this, &_image_matching_with_UI::applyTownParams);
    presetLayout->addWidget(btnHammer);
    presetLayout->addWidget(btnTown);
    controlLayout->addLayout(presetLayout);

    // 4. 操作按钮与进度条
    QGroupBox* grpRun = new QGroupBox("运行控制");
    QVBoxLayout* runLayout = new QVBoxLayout(grpRun);

    // 状态说明标签
    lblProgressStatus = new QLabel("准备就绪");
    lblProgressStatus->setAlignment(Qt::AlignCenter);
    lblProgressStatus->setStyleSheet("color: #666666; font-style: italic; margin-bottom: 2px;");

    // 进度条
    progressBar = new QProgressBar();
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setTextVisible(true);
    progressBar->setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #4CAF50; }");

    QPushButton* btnRun = new QPushButton("开始运行匹配");
    btnRun->setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; background-color: #2196F3; color: white;");
    connect(btnRun, &QPushButton::clicked, this, &_image_matching_with_UI::runMatching);

    runLayout->addWidget(btnRun);
    runLayout->addWidget(lblProgressStatus); // 添加状态标签
    runLayout->addWidget(progressBar);
    controlLayout->addWidget(grpRun);

    // 5. 结果输出设置
    QGroupBox* grpSave = new QGroupBox("结果输出设置");
    QGridLayout* saveLayout = new QGridLayout(grpSave);

    chkAutoSaveAfterRun = new QCheckBox("运行完成后自动保存");
    chkAutoSaveAfterRun->setChecked(true); // 默认启用自动保存

    chkSaveToResDir = new QCheckBox("默认保存到 ./res 文件夹"); // 路径选项
    chkSaveToResDir->setChecked(true);     // 默认启用

    QLabel* lblName = new QLabel("保存文件名:");
    saveFileNameEdit = new QLineEdit();
    saveFileNameEdit->setPlaceholderText("Left_Right");

    QPushButton* btnSave = new QPushButton("手动保存结果");
    connect(btnSave, &QPushButton::clicked, this, &_image_matching_with_UI::saveResults);

    // 布局调整
    // Row 0: 两个复选框
    saveLayout->addWidget(chkAutoSaveAfterRun, 0, 0, 1, 2);
    saveLayout->addWidget(chkSaveToResDir, 1, 0, 1, 2); // 新增的一行

    // Row 2: 文件名输入
    saveLayout->addWidget(lblName, 2, 0);
    saveLayout->addWidget(saveFileNameEdit, 2, 1);

    // Row 3: 保存按钮
    saveLayout->addWidget(btnSave, 3, 0, 1, 2);

    controlLayout->addWidget(grpSave);
    controlLayout->addStretch();

    // --- 连接信号以自动更新文件名 ---
    // 当文件路径改变时，尝试更新默认保存文件名
    connect(pathLeftEdit, &QLineEdit::textChanged, this, &_image_matching_with_UI::updateDefaultFilename);
    connect(pathRightEdit, &QLineEdit::textChanged, this, &_image_matching_with_UI::updateDefaultFilename);

    // --- 右侧：结果展示 ---
    QTabWidget* tabs = new QTabWidget();

    // Tab 1: 可视化
    QWidget* tabVis = new QWidget();
    QVBoxLayout* visLayout = new QVBoxLayout(tabVis);
    imageView = new ZoomGraphicsView(this);
    imageView->setScene(imageScene);

    // --- 自动检测系统主题颜色 ---
    // 获取当前应用程序的调色板
    QPalette pal = QApplication::palette();

    // 获取文本亮度和背景亮度
    int textLightness = pal.color(QPalette::WindowText).lightness();
    int bgLightness = pal.color(QPalette::Window).lightness();

    // 判断逻辑：如果文本比背景亮，说明是深色模式
    bool isDarkMode = textLightness > bgLightness;

    // 根据模式设置背景色：深色模式用 40，浅色模式用 240
    QColor targetBgColor = isDarkMode ? QColor(40, 40, 40) : QColor(240, 240, 240);

    imageView->setBackgroundBrush(QBrush(targetBgColor));

    visLayout->addWidget(imageView);
    tabs->addTab(tabVis, "可视化结果");

    // Tab 2: 精度报告
    reportEdit = new QTextEdit();
    reportEdit->setReadOnly(true);
    tabs->addTab(reportEdit, "精度报告");

    // Tab 3: 匹配点列表
    pointsTable = new QTableWidget();
    pointsTable->setColumnCount(5);
    pointsTable->setHorizontalHeaderLabels({ "ID", "L_X", "L_Y", "R_X", "R_Y" });
    tabs->addTab(pointsTable, "点位数据");

    splitter->addWidget(controlPanel);
    splitter->addWidget(tabs);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 5);

    mainLayout->addWidget(splitter);
}

// --- 槽函数实现 ---

void _image_matching_with_UI::selectLeftImage() {
    QString f = QFileDialog::getOpenFileName(this, "选择左影像", "", "Images (*.jpg *.png *.bmp *.tif)");
    if (!f.isEmpty()) pathLeftEdit->setText(f);
}

void _image_matching_with_UI::selectRightImage() {
    QString f = QFileDialog::getOpenFileName(this, "选择右影像", "", "Images (*.jpg *.png *.bmp *.tif)");
    if (!f.isEmpty()) pathRightEdit->setText(f);
}

void _image_matching_with_UI::applyHammerParams() {
    pathLeftEdit->setText("dataset/1-hammer/hammerL.jpg");
    pathRightEdit->setText("dataset/1-hammer/hammerR.jpg");
    spinMoWin->setValue(7); spinMoT->setValue(1200); spinMoNms->setValue(50);
    spinOffX->setValue(1895); spinOffY->setValue(0);
    spinSearch->setValue(25); spinMatch->setValue(9);
    spinLsmWin->setValue(25);
}

void _image_matching_with_UI::applyTownParams() {
    pathLeftEdit->setText("dataset/2-town/l.jpg");
    pathRightEdit->setText("dataset/2-town/r.jpg");
    spinMoWin->setValue(7); spinMoT->setValue(1600); spinMoNms->setValue(61);
    spinOffX->setValue(122); spinOffY->setValue(34);
    spinSearch->setValue(21); spinMatch->setValue(7);
    spinLsmWin->setValue(21);
}

void _image_matching_with_UI::runMatching() {
    QString lPath = pathLeftEdit->text();
    QString rPath = pathRightEdit->text();

    Mat imgL = imread(lPath.toStdString());
    Mat imgR = imread(rPath.toStdString());

    if (imgL.empty() || imgR.empty()) {
        QMessageBox::critical(this, "错误", "无法读取图像文件，请检查路径。");
        return;
    }

    // 初始化界面状态
    reportEdit->clear();
    progressBar->setValue(0);
    lblProgressStatus->setText("初始化中...");

    appendReport("=== 开始处理 ===");
    appendReport("左图: " + lPath);
    appendReport("右图: " + rPath);

    QApplication::processEvents();

    stringstream ss;

    // ------------------------------------------------
    // [Step 1] 特征提取
    // ------------------------------------------------
    progressBar->setValue(5);
    lblProgressStatus->setText("步骤 1/4: 特征提取 (Shi-Tomasi)...");
    appendReport("正在进行特征提取 (Shi-Tomasi)...");

    // 强制刷新UI以显示文本
    QApplication::processEvents();

    vector<Point2f> kpts;
    ExtractFeaturesShiTomasi(imgL, kpts, spinMoWin->value(), spinMoNms->value(), spinMoT->value());

    ss << "[Step 1] 特征提取完成: " << kpts.size() << " 点\n";
    appendReport(QString::fromStdString(ss.str())); ss.str("");
    progressBar->setValue(30);

    // ------------------------------------------------
    // [Step 2] NCC 粗匹配
    // ------------------------------------------------
    lblProgressStatus->setText("步骤 2/4: NCC 粗匹配...");
    appendReport("正在进行 NCC 粗匹配...");
    QApplication::processEvents();

    vector<matchPoint> rhoMatches;
    rhoMatchImproved(imgL, imgR, kpts, rhoMatches,
        spinOffX->value(), spinOffY->value(), spinMatch->value(), spinSearch->value());

    ss << "[Step 2] NCC 粗匹配完成: " << rhoMatches.size() << " 点\n";
    evaluateEpipolarGeometry(rhoMatches, "NCC Stage", ss);
    appendReport(QString::fromStdString(ss.str())); ss.str("");
    progressBar->setValue(60);

    // ------------------------------------------------
    // [Step 3] LSM 精匹配
    // ------------------------------------------------
    lblProgressStatus->setText("步骤 3/4: LSM 最小二乘精匹配...");
    appendReport("正在进行 LSM 最小二乘匹配...");
    QApplication::processEvents();

    vector<matchPoint> lsmMatches;
    LSM_Improved(imgL, imgR, rhoMatches, lsmMatches, spinLsmWin->value(), 0.001, 30);

    ss << "[Step 3] LSM 精匹配完成: " << lsmMatches.size() << " 点\n";
    evaluateEpipolarGeometry(lsmMatches, "LSM Stage", ss);
    appendReport(QString::fromStdString(ss.str()));
    progressBar->setValue(90);

    // ------------------------------------------------
    // [Step 4] 结果整理与显示
    // ------------------------------------------------
    lblProgressStatus->setText("步骤 4/4: 生成可视化结果...");
    appendReport("正在更新可视化视图...");
    QApplication::processEvents();

    // 缓存结果
    lastMatches = lsmMatches;
    lastImgL = imgL;
    lastImgR = imgR;

    updateTable(lsmMatches);
    Mat vis = drawMatchesVisual(imgL, imgR, lsmMatches);
    displayImage(vis);

    progressBar->setValue(100);
    lblProgressStatus->setText("处理完成");
    appendReport("=== 匹配流程结束 ===");

    // 检查是否自动保存
    if (chkAutoSaveAfterRun->isChecked()) {
        lblProgressStatus->setText("正在自动保存...");
        appendReport("正在执行自动保存...");
        saveResults();
        lblProgressStatus->setText("已完成");
    }
    else {
        QMessageBox::information(this, "完成", "匹配处理完成！");
    }
}

void _image_matching_with_UI::saveResults() {
    if (lastMatches.empty()) {
        QMessageBox::warning(this, "警告", "没有可保存的结果，请先运行匹配。");
        return;
    }

    // 确定保存目录：根据勾选状态决定是否使用 ./res
    QString dirPath = QDir::currentPath();
    if (chkSaveToResDir->isChecked()) {
        dirPath += "/res";
    }

    QDir dir(dirPath);
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    // 获取用户输入的文件名
    QString userFileName = saveFileNameEdit->text().trimmed();
    if (userFileName.isEmpty()) {
        userFileName = "Result"; // 兜底默认名
    }

    // 构建完整路径前缀
    string filePrefix = "/" + userFileName.toStdString();

    string visPath = dirPath.toStdString() + filePrefix + "_Vis.jpg";
    string ptsPath = dirPath.toStdString() + filePrefix + "_Points.txt";
    string repPath = dirPath.toStdString() + filePrefix + "_Report.txt";

    // 保存图片
    Mat vis = drawMatchesVisual(lastImgL, lastImgR, lastMatches);
    imwrite(visPath, vis);

    // 保存点位
    ofstream out(ptsPath);
    out << "ID\tXL\tYL\tXR\tYR\n";
    for (size_t i = 0; i < lastMatches.size(); i++) {
        out << (i + 1) << "\t" << fixed << setprecision(4)
            << lastMatches[i].xl << "\t" << lastMatches[i].yl << "\t"
            << lastMatches[i].xr << "\t" << lastMatches[i].yr << "\n";
    }
    out.close();

    // 保存报告
    ofstream rep(repPath);
    rep << reportEdit->toPlainText().toStdString();
    rep.close();

    // 只有在手动点击保存时（或者非自动流程中需要反馈时）才考虑是否弹窗
    // 为了体验流畅，只在日志中打印路径
    appendReport("结果已保存至: " + dirPath);
}

// --- 辅助实现 ---

QSpinBox* _image_matching_with_UI::createSpinBox(int min, int max, int val, const QString& tooltip) {
    QSpinBox* sb = new QSpinBox();
    sb->setRange(min, max);
    sb->setValue(val);
    sb->setToolTip(tooltip);
    return sb;
}

void _image_matching_with_UI::appendReport(const QString& text) {
    reportEdit->append(text);
    // 强制刷新界面事件循环，实现“实时日志”效果
    QApplication::processEvents();
}

void _image_matching_with_UI::updateTable(const vector<matchPoint>& points) {
    pointsTable->setRowCount(0);
    pointsTable->setRowCount((int)points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        pointsTable->setItem(i, 0, new QTableWidgetItem(QString::number(i + 1)));
        pointsTable->setItem(i, 1, new QTableWidgetItem(QString::number(points[i].xl, 'f', 3)));
        pointsTable->setItem(i, 2, new QTableWidgetItem(QString::number(points[i].yl, 'f', 3)));
        pointsTable->setItem(i, 3, new QTableWidgetItem(QString::number(points[i].xr, 'f', 3)));
        pointsTable->setItem(i, 4, new QTableWidgetItem(QString::number(points[i].yr, 'f', 3)));
    }
}

void _image_matching_with_UI::displayImage(const Mat& img) {
    if (img.empty()) return;

    Mat temp;
    if (img.channels() == 3) cvtColor(img, temp, COLOR_BGR2RGB);
    else cvtColor(img, temp, COLOR_GRAY2RGB);

    QImage qimg(temp.data, temp.cols, temp.rows, (int)temp.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qimg.copy());

    imageScene->clear();
    imageView->resetTransform();
    imageScene->addPixmap(pixmap);
    imageScene->setSceneRect(pixmap.rect());
    imageView->fitInView(imageScene->itemsBoundingRect(), Qt::KeepAspectRatio);
}

// --- 核心算法部分 ---

void _image_matching_with_UI::ExtractFeaturesShiTomasi(const Mat& imgBGRorGray, vector<Point2f>& outPts, int blockSizeHint, int minDistHint, double qualityHint) {
    outPts.clear();
    if (imgBGRorGray.empty()) return;
    Mat gray;
    if (imgBGRorGray.channels() == 3) cvtColor(imgBGRorGray, gray, COLOR_BGR2GRAY);
    else gray = imgBGRorGray;

    int maxCorners = 4000;
    int blockSize = std::max(3, blockSizeHint | 1);
    double qualityLevel = 0.01;
    if (qualityHint > 0.0) {
        qualityLevel = std::clamp(qualityHint / 100000.0, 0.004, 0.03);
    }
    double minDistance = std::max(3, minDistHint / 2);
    bool useHarrisDetector = false;
    double k = 0.04;

    goodFeaturesToTrack(gray, outPts, maxCorners, qualityLevel, minDistance,
        noArray(), blockSize, useHarrisDetector, k);
}

void _image_matching_with_UI::rhoMatchImproved(const Mat& imgL, const Mat& imgR,
    const vector<Point2f>& keypointsL,
    vector<matchPoint>& outMatches,
    int offsetX, int offsetY, int matchSize, int searchSize) {

    // 硬编码 Console 版本中的参数
    double rhoThresh = 0.80;
    double marginThresh = 0.03;
    int crossCheckTol = 1;

    outMatches.clear();
    if (imgL.empty() || imgR.empty()) return;

    Mat L, R;
    if (imgL.channels() == 3) cvtColor(imgL, L, COLOR_BGR2GRAY);
    else L = imgL;
    if (imgR.channels() == 3) cvtColor(imgR, R, COLOR_BGR2GRAY);
    else R = imgR;

    const int halfMatch = matchSize / 2;
    const int halfSearch = searchSize / 2;

    for (const auto& pt : keypointsL) {
        int xl = (int)std::lround(pt.x);
        int yl = (int)std::lround(pt.y);

        if (!patchInside(L, xl, yl, halfMatch)) continue;

        int xr_est = xl - offsetX;
        int yr_est = yl - offsetY;

        // 越界检查
        if (xr_est - halfSearch - halfMatch < 0 || yr_est - halfSearch - halfMatch < 0 ||
            xr_est + halfSearch + halfMatch >= R.cols || yr_est + halfSearch + halfMatch >= R.rows) {
            continue;
        }

        // L->R
        NCCBest2 lr = searchBest2NCC(L, R, xl, yl, xr_est, yr_est, halfMatch, halfSearch);
        if (!lr.ok) continue;
        if (lr.bestRho < rhoThresh) continue;
        if ((lr.bestRho - lr.secondRho) < marginThresh) continue;

        // R->L
        int xr = lr.bestPt.x;
        int yr = lr.bestPt.y;
        int xl_est_back = xr + offsetX;
        int yl_est_back = yr + offsetY;

        if (xl_est_back - halfSearch - halfMatch < 0 || yl_est_back - halfSearch - halfMatch < 0 ||
            xl_est_back + halfSearch + halfMatch >= L.cols || yl_est_back + halfSearch + halfMatch >= L.rows) {
            continue;
        }

        // 回溯搜索：固定 R 中的最佳点，在 L 中搜索
        NCCBest2 rl;
        for (int dy = -halfSearch; dy <= halfSearch; ++dy) {
            for (int dx = -halfSearch; dx <= halfSearch; ++dx) {
                int xli = xl_est_back + dx;
                int yli = yl_est_back + dy;
                double rho = -1e9;
                if (!computeNCC(L, R, xli, yli, xr, yr, halfMatch, rho)) continue;
                if (rho > rl.bestRho) {
                    rl.secondRho = rl.bestRho;
                    rl.bestRho = rho;
                    rl.bestPt = Point(xli, yli);
                    rl.ok = true;
                }
                else if (rho > rl.secondRho) {
                    rl.secondRho = rho;
                }
            }
        }
        if (!rl.ok) continue;
        if (rl.bestRho < rhoThresh) continue;
        if ((rl.bestRho - rl.secondRho) < marginThresh) continue;

        if (std::abs(rl.bestPt.x - xl) > crossCheckTol || std::abs(rl.bestPt.y - yl) > crossCheckTol) {
            continue;
        }

        matchPoint mp;
        mp.xl = (double)xl;
        mp.yl = (double)yl;
        mp.xr = (double)xr;
        mp.yr = (double)yr;
        outMatches.push_back(mp);
    }
}

void _image_matching_with_UI::LSM_Improved(const Mat& imgL, const Mat& imgR,
    const vector<matchPoint>& inMatches,
    vector<matchPoint>& outRefined,
    int winSize,
    double lambda,
    int maxIters)
{
    outRefined.clear();
    if (imgL.empty() || imgR.empty()) return;

    Mat L, R;
    if (imgL.channels() == 3) cvtColor(imgL, L, COLOR_BGR2GRAY);
    else L = imgL;
    if (imgR.channels() == 3) cvtColor(imgR, R, COLOR_BGR2GRAY);
    else R = imgR;

    Mat Rf; R.convertTo(Rf, CV_64F);
    Mat gradX, gradY;
    Sobel(Rf, gradX, CV_64F, 1, 0, 3);
    Sobel(Rf, gradY, CV_64F, 0, 1, 3);

    const int halfWin = winSize / 2;
    const int N = winSize * winSize;

    for (const auto& mp0 : inMatches) {
        double xl0_f = mp0.xl;
        double yl0_f = mp0.yl;
        double xr0 = mp0.xr;
        double yr0 = mp0.yr;

        int xl_int = (int)std::round(xl0_f);
        int yl_int = (int)std::round(yl0_f);

        if (!patchInside(L, xl_int, yl_int, halfWin)) continue;

        // 初始参数 (Identity)
        double p_a0 = 0.0, p_a1 = 0.0, p_a2 = 1.0; // Y方向
        double p_b0 = 0.0, p_b1 = 1.0, p_b2 = 0.0; // X方向

        bool ok = true;
        vector<double> valsL(N), valsR(N);
        Mat C = Mat::zeros(N, 6, CV_64F);
        Mat Lvec = Mat::zeros(N, 1, CV_64F);

        for (int iter = 0; iter < maxIters; ++iter) {
            int idx = 0;

            // 采样与构建矩阵
            for (int i = 0; i < winSize; ++i) {
                double v = (double)i - (double)halfWin;
                for (int j = 0; j < winSize; ++j, ++idx) {
                    double u = (double)j - (double)halfWin;

                    // 右图采样 (Affine)
                    double xw = xr0 + p_b0 + p_b1 * u + p_b2 * v;
                    double yw = yr0 + p_a0 + p_a1 * u + p_a2 * v;

                    // 左图采样 (Identity)
                    double xl = xl0_f + u;
                    double yl = yl0_f + v;

                    if (!insideForBilinear(R.cols, R.rows, xw, yw) ||
                        !insideForBilinear(gradX.cols, gradX.rows, xw, yw) ||
                        !insideForBilinear(gradY.cols, gradY.rows, xw, yw) ||
                        !insideForBilinear(L.cols, L.rows, xl, yl)) {
                        ok = false; break;
                    }

                    double gR = bilinearU8(R, xw, yw);
                    double gL = bilinearU8(L, xl, yl);

                    valsR[idx] = gR;
                    valsL[idx] = gL;

                    double gRx = bilinearF64(gradX, xw, yw);
                    double gRy = bilinearF64(gradY, xw, yw);

                    // 构造 Jacobian (注意索引顺序与 Console 一致)
                    // 几何参数 Y (a0, a1, a2) -> 对应索引 0, 1, 2
                    C.at<double>(idx, 0) = gRy * 1.0;
                    C.at<double>(idx, 1) = gRy * u;
                    C.at<double>(idx, 2) = gRy * v;

                    // 几何参数 X (b0, b1, b2) -> 对应索引 3, 4, 5
                    C.at<double>(idx, 3) = gRx * 1.0;
                    C.at<double>(idx, 4) = gRx * u;
                    C.at<double>(idx, 5) = gRx * v;
                }
                if (!ok) break;
            }
            if (!ok) break;

            double meanL = getPatchMean(valsL);
            double meanR = getPatchMean(valsR);
            double diffVal = meanL - meanR;

            for (int k = 0; k < N; ++k) {
                Lvec.at<double>(k, 0) = valsL[k] - (valsR[k] + diffVal);
            }

            Mat dx;
            if (!solveDampedLeastSquares(C, Lvec, dx, lambda)) { ok = false; break; }

            p_a0 += dx.at<double>(0, 0);
            p_a1 += dx.at<double>(1, 0);
            p_a2 += dx.at<double>(2, 0);

            p_b0 += dx.at<double>(3, 0);
            p_b1 += dx.at<double>(4, 0);
            p_b2 += dx.at<double>(5, 0);

            if (std::abs(p_b1 - 1.0) > 0.3 || std::abs(p_a2 - 1.0) > 0.3 ||
                std::abs(p_b2) > 0.3 || std::abs(p_a1) > 0.3) {
                ok = false; break;
            }

            double maxDelta = 0.0;
            for (int k = 0; k < 6; ++k) maxDelta = std::max(maxDelta, std::abs(dx.at<double>(k, 0)));
            if (maxDelta < 1e-4) break;
        }

        if (!ok) continue;

        matchPoint mp;
        mp.xl = xl0_f;
        mp.yl = yl0_f;
        mp.xr = xr0 + p_b0;
        mp.yr = yr0 + p_a0;

        // Outlier Rejection
        double shiftX = std::abs(mp.xr - mp0.xr);
        double shiftY = std::abs(mp.yr - mp0.yr);

        // 剔除阈值 (与 Console 保持一致)
        if (shiftY > 1.5 || shiftX > 2.0) {
            continue;
        }

        if (mp.xr < 0.0 || mp.yr < 0.0 || mp.xr >= imgR.cols || mp.yr >= imgR.rows) continue;
        outRefined.push_back(mp);
    }
}

void _image_matching_with_UI::evaluateEpipolarGeometry(const vector<matchPoint>& points, const string& stageName, ostream& out) {
    out << "--- [" << stageName << "] 核线精度评定 (F-Matrix + RANSAC) ---\n";

    if (points.size() < 8) {
        out << "匹配点数量不足 8 个，无法计算基础矩阵。\n\n";
        return;
    }

    vector<Point2f> ptsL, ptsR;
    ptsL.reserve(points.size());
    ptsR.reserve(points.size());
    for (const auto& mp : points) {
        ptsL.push_back(Point2f((float)mp.xl, (float)mp.yl));
        ptsR.push_back(Point2f((float)mp.xr, (float)mp.yr));
    }

    vector<uchar> mask;
    Mat F = findFundamentalMat(ptsL, ptsR, FM_RANSAC, 3.0, 0.99, mask);

    if (F.empty()) {
        out << "计算基础矩阵失败 (F is empty)。\n\n";
        return;
    }

    Mat F64;
    F.convertTo(F64, CV_64F);

    double sumSqErr = 0.0;
    double sumErr = 0.0;
    int inlierCount = 0;
    double maxErr = 0.0;

    for (size_t i = 0; i < points.size(); ++i) {
        if (!mask[i]) continue;

        double x1 = ptsL[i].x;
        double y1 = ptsL[i].y;
        double x2 = ptsR[i].x;
        double y2 = ptsR[i].y;

        Mat p1 = (Mat_<double>(3, 1) << x1, y1, 1.0);
        Mat l2 = F64 * p1;
        double a = l2.at<double>(0, 0);
        double b = l2.at<double>(1, 0);
        double c = l2.at<double>(2, 0);

        double num = std::abs(a * x2 + b * y2 + c);
        double den = std::sqrt(a * a + b * b);
        double d = (den > 1e-8) ? (num / den) : 0.0;

        sumErr += d;
        sumSqErr += (d * d);
        maxErr = std::max(maxErr, d);
        inlierCount++;
    }

    if (inlierCount == 0) {
        out << "未找到满足 RANSAC 的内点。\n\n";
        return;
    }

    double meanErr = sumErr / (double)inlierCount;
    double rmse = std::sqrt(sumSqErr / (double)inlierCount);
    double inlierRatio = 100.0 * (double)inlierCount / (double)points.size();

    out << "输入点数: " << points.size() << "\n";
    out << "RANSAC 内点数: " << inlierCount << " (" << fixed << setprecision(2) << inlierRatio << "%)\n";
    out << "平均核线误差 (Mean Error): " << setprecision(4) << meanErr << " pixels\n";
    out << "核线误差均方根 (RMSE):     " << setprecision(4) << rmse << " pixels\n";
    out << "最大误差 (Max Error):      " << setprecision(4) << maxErr << " pixels\n";
    out << "(注: 仅统计 RANSAC 内点误差，阈值=3.0px)\n\n";
}

Mat _image_matching_with_UI::drawMatchesVisual(const Mat& imgL, const Mat& imgR, const vector<matchPoint>& points) {
    int h = max(imgL.rows, imgR.rows);
    int w = imgL.cols + imgR.cols;
    Mat canvas(h, w, CV_8UC3, Scalar(0, 0, 0));

    Mat roiL = canvas(Rect(0, 0, imgL.cols, imgL.rows));
    if (imgL.channels() == 3) imgL.copyTo(roiL);
    else cvtColor(imgL, roiL, COLOR_GRAY2BGR);

    Mat roiR = canvas(Rect(imgL.cols, 0, imgR.cols, imgR.rows));
    if (imgR.channels() == 3) imgR.copyTo(roiR);
    else cvtColor(imgR, roiR, COLOR_GRAY2BGR);

    double maxDim = (double)std::max(w, h);
    double scale = maxDim / 1200.0;
    if (scale < 0.5) scale = 0.5;

    int lineThick = std::max(1, (int)(1.5 * scale));
    int ptRadius = std::max(2, (int)(4.0 * scale));

    RNG rng(12345);

    for (const auto& pt : points) {
        Point2f p1(pt.xl, pt.yl);
        Point2f p2(pt.xr + (double)imgL.cols, pt.yr);

        int hue = rng.uniform(0, 180);
        Mat hsv(1, 1, CV_8UC3, Scalar(hue, 255, 255));
        Mat bgr;
        cvtColor(hsv, bgr, COLOR_HSV2BGR);
        Scalar color = Scalar(bgr.at<Vec3b>(0, 0));

        line(canvas, p1, p2, color, lineThick, LINE_AA);
        circle(canvas, p1, ptRadius, color, lineThick, LINE_AA);
        circle(canvas, p2, ptRadius, color, lineThick, LINE_AA);
    }
    return canvas;
}

void _image_matching_with_UI::updateDefaultFilename() {
    QFileInfo fiL(pathLeftEdit->text());
    QFileInfo fiR(pathRightEdit->text());

    QString baseNameL = fiL.baseName();
    QString baseNameR = fiR.baseName();

    if (baseNameL.isEmpty()) baseNameL = "Left";
    if (baseNameR.isEmpty()) baseNameR = "Right";

    // 设置默认文件名，例如 "hammerL_hammerR"
    // 只有当用户没有手动修改过（或者当前为空）时才覆盖，体验更好
    // 这里简化逻辑：只要路径变了就更新推荐名
    QString defaultName = baseNameL + "_" + baseNameR;
    saveFileNameEdit->setText(defaultName);
}
