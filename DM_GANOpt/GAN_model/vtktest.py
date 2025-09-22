# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:58:28 2024

@author: sneve
"""

import vtk

# Simple VTK script to verify installation
sphere = vtk.vtkSphereSource()
sphere.SetRadius(5.0)
sphere.SetPhiResolution(50)
sphere.SetThetaResolution(50)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(sphere.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1, 1, 1)  # White background

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

render_window.Render()
render_window_interactor.Start()