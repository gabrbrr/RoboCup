#include "ObstaclesOrientation.h"
#include "Tools/Debugging/DebugDrawings.h"

void ObstaclesOrientation::draw() const
{
  DEBUG_DRAWING("representation:ObstaclesOrientation:image", "drawingOnImage")
  {
    for(const auto& obstacle : obstacles)
      RECTANGLE("representation:ObstaclesOrientation:image", obstacle.left, obstacle.top, obstacle.right, obstacle.bottom, 4, Drawings::solidPen, ColorRGBA::white);
  }
}
