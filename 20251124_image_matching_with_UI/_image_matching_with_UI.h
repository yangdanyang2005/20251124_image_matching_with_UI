#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGraphicsScene>
#include <QtWidgets/QGraphicsPixmapItem>
#include <QWheelEvent>
#include <QtWidgets/QProgressBar>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ui__image_matching_with_UI.h"

using namespace std;
using namespace cv;

struct matchPoint {
    double xl, yl, xr, yr;
};

// ==========================================
// 自定义视图类：支持鼠标滚轮缩放
// ==========================================
class ZoomGraphicsView : public QGraphicsView {
public:
    ZoomGraphicsView(QWidget* parent = nullptr) : QGraphicsView(parent) {
        // 以此启用鼠标拖拽移动图片
        setDragMode(QGraphicsView::ScrollHandDrag);
        // 去除背景噪点，使用平滑变换
        setRenderHint(QPainter::Antialiasing);
        setRenderHint(QPainter::SmoothPixmapTransform);
    }

protected:
    // 覆写滚轮事件实现缩放
    void wheelEvent(QWheelEvent* event) override {
        // 获取滚轮滚动的角度
        int angle = event->angleDelta().y();
        double factor;

        if (angle > 0) {
            factor = 1.1; // 放大 10%
        }
        else {
            factor = 0.9; // 缩小 10%
        }

        // 执行缩放
        scale(factor, factor);
    }
};

class _image_matching_with_UI : public QMainWindow
{
    Q_OBJECT

public:
    _image_matching_with_UI(QWidget* parent = Q_NULLPTR);
    ~_image_matching_with_UI();

private slots:
    void selectLeftImage();
    void selectRightImage();
    void applyHammerParams();
    void applyTownParams();
    void runMatching();
    void saveResults();
    void updateDefaultFilename();

private:
    Ui::_image_matching_with_UIClass ui;

    QLineEdit* pathLeftEdit, * pathRightEdit;
    QCheckBox* chkAutoSaveAfterRun; // "运行完成后自动保存"
    QCheckBox* chkSaveToResDir;     // "默认保存到 ./res 文件夹"
    QLineEdit* saveFileNameEdit;    // "保存文件名"
    QLabel* lblProgressStatus;      // 进度状态说明标签
    QProgressBar* progressBar;      // 进度条
    QSpinBox* spinMoWin, * spinMoNms, * spinOffX, * spinOffY, * spinSearch, * spinMatch, * spinLsmWin;
    QDoubleSpinBox* spinMoT;
    QTextEdit* reportEdit;
    QTableWidget* pointsTable;

    ZoomGraphicsView* imageView; // 自定义视图
    QGraphicsScene* imageScene;  // 场景容器

    vector<matchPoint> lastMatches;
    Mat lastImgL, lastImgR;

    void setupCustomUI();
    QSpinBox* createSpinBox(int min, int max, int val, const QString& tooltip);
    void appendReport(const QString& text);
    void updateTable(const vector<matchPoint>& points);

    void displayImage(const Mat& img);

    // --- 核心算法函数 ---
    void ExtractFeaturesShiTomasi(const Mat& imgBGRorGray, vector<Point2f>& outPts, int blockSizeHint, int minDistHint, double qualityHint);

    void rhoMatchImproved(const Mat& imgL, const Mat& imgR, const vector<Point2f>& keypointsL, vector<matchPoint>& outMatches, int offsetX, int offsetY, int matchSize, int searchSize);

    void LSM_Improved(const Mat& imgL, const Mat& imgR, const vector<matchPoint>& inMatches, vector<matchPoint>& outRefined, int winSize, double lambda, int maxIters);

    void evaluateEpipolarGeometry(const vector<matchPoint>& points, const string& stageName, ostream& out);
    Mat drawMatchesVisual(const Mat& imgL, const Mat& imgR, const vector<matchPoint>& points);
};