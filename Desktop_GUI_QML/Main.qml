import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: root
    width: 1280
    height: 800
    visible: true
    title: qsTr("ChessPlayerAnalyzer")
    color: "#121212"

    property color accent: "#3eb29b" // color de acento teal

    StackView {
        id: stack
        anchors.fill: parent
        initialItem: HomePage { stack: stack }
    }
}
